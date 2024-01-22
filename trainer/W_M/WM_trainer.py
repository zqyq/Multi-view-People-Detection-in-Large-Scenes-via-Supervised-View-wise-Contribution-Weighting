import time
import torch
import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from evaluation.evaluate import evaluate
from utils.nms import nms
from utils.meters import AverageMeter
from utils.image_utils import add_heatmap_to_image, vertical_prediction_vis
from utils.gaussian_mse import target_transform
from utils.person_help import vis


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PerspectiveTrainer:
    def __init__(self, model, args, logdir, denormalize, **kwargs):
        self.model = model
        self.args = args

        self.weight_2D = args.weight_2D
        self.weight_svp = args.weight_svp

        self.fix_2D = args.fix_2D
        self.fix_svp = args.fix_svp
        self.fix_weight = args.fix_weight

        self.cls_thres = args.cls_thres
        self.nms_thres = args.nms_thres
        self.dist_thres = args.dist_thres
        self.logdir = logdir
        self.denormalize = denormalize
        self.num_cam = model.num_cam

    def test(self, data_loader, res_fpath):
        print('Testing...')
        self.model.eval()
        losses = 0
        all_res_list = []
        t0 = time.time()

        for batch_idx, (imgs, gp_gt, imgs_gt, frame) in enumerate(data_loader):
            with torch.no_grad():
                img_res, view_gp_res, gp_res, mask = self.model(imgs)
            if res_fpath is not None:
                map_grid_res = gp_res.detach().cpu().squeeze()
                map_grid_res = torch.relu(map_grid_res)
                map_grid_res = (map_grid_res - map_grid_res.min()) / (map_grid_res.max() - map_grid_res.min() + 1e-12)
                v_s = map_grid_res[map_grid_res > self.cls_thres].unsqueeze(1)
                grid_ij = (map_grid_res > self.cls_thres).nonzero()
                if data_loader.dataset.base.indexing == 'xy':
                    grid_xy = grid_ij[:, [1, 0]]
                else:
                    grid_xy = grid_ij
                all_res_list.append(torch.cat([torch.ones_like(v_s) * frame, grid_xy.float() *
                                               data_loader.dataset.grid_reduce, v_s], dim=1))

            # img_2D loss
            loss_2D = F.mse_loss(img_res, imgs_gt[0].to(img_res.device)) / data_loader.dataset.num_cam
            # SVP loss
            view_gp_gt = gp_gt.to(mask.device) * mask / data_loader.dataset.num_cam
            loss_svp = F.mse_loss(view_gp_res, view_gp_gt)
            # loss fusion
            loss_fusion = F.mse_loss(gp_res, gp_gt.to(gp_res.device))

            loss = loss_fusion + self.weight_svp * loss_svp + self.weight_2D * loss_2D
            losses += loss.item()

            val_interval = len(data_loader) // 4
            if (batch_idx + 1) % val_interval == 0:
                # visulaization in training process
                # 2D image
                heatmap0_head = img_res[0].detach().cpu().numpy().squeeze()
                img0 = self.denormalize(imgs[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
                img0 = Image.fromarray((img0 * 255).astype('uint8'))
                head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
                head_cam_result.save(os.path.join(self.logdir, f'b_{batch_idx + 1}_cam1_head.jpg'))
                # single-view prediction
                vertical_prediction_vis(view_gp_res[0], view_gp_gt[0],
                               save_dir=os.path.join(self.logdir, f'b_{batch_idx + 1}_cam1_gp_res.jpg'))
                # ground plane fusion prediction
                vertical_prediction_vis(gp_res, gp_gt, save_dir=os.path.join(self.logdir, f'b_{batch_idx + 1}_gp_res.jpg'))

        t1 = time.time()
        t_epoch = t1 - t0
        print(
            f'################################### Testing #############################################\n'
            f'Batch:{len(data_loader)}, Loss:{losses / len(data_loader):.6f} [2D {loss_2D.item():.6f},'
            f' SVP {loss_svp.item():.6f}, fusion {loss_fusion.item():.6f},], Time:{t_epoch:.1f} '
            f' maxima:{gp_res.max():.3f}')
        moda = 0
        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list.numpy(), '%.8f')
            res_list = []
            for frame in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, 20, np.inf)
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            np.savetxt(res_fpath, res_list, '%d')

            recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), data_loader.dataset.gt_fpath,
                                                     20, data_loader.dataset.base.__name__)
            F1_score = 2 * precision * recall / (precision + recall + 1e-12)
            print(f'moda: {moda:.1f}%, modp: {modp:.1f}%,'
                  f' precision: {precision:.1f}%, recall: {recall:.1f}%, F1_score:{F1_score:.1f}%')

  