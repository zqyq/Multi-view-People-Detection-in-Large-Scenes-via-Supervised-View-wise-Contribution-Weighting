import os
import time
# import tqdm
import torch
# import torch.optim as optim
import torch.nn.functional as F
import tqdm

from PIL import Image
import numpy as np
from utils.nms import nms
from evaluation.evaluate import evaluate
from utils.person_help import vis
import matplotlib.pyplot as plt
from utils.image_utils import add_heatmap_to_image, prediction_vis


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
        self.metrics = {'moda': 0, 'modp': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}

    def train(self, variant, data_loader, epoch, optimizer, scheduler):
        if variant == '2D':
            self.train_2D(data_loader, epoch, optimizer, scheduler)
        elif variant == '2D_SVP':
            self.train_2D_SVP(data_loader, epoch, optimizer, scheduler)
        elif variant == '2D_SVP_VCW':
            self.train_2D_SVP_VCW(data_loader, epoch, optimizer, scheduler)
        else:
            raise Exception("Wrong variant.")
        # save
        checkpoint = {'model': self.model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'scheduler': scheduler.state_dict()}
        torch.save(checkpoint, os.path.join(self.logdir, f'latest_{variant}_model.pth'))
        if isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR) or\
                isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
            scheduler.step()
    def train_2D(self, data_loader, epoch, optimizer, scheduler):
        print("####################  Training 2D model  ########################")
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0
        losses = 0
        for batch_idx, (imgs, imgs_gt, view_gp_gt, gp_gt, frame) in enumerate(data_loader):
            imgs_gt = imgs_gt.permute(1, 0, 2, 3)
            # view_gp_gt = view_gp_gt.permute(1, 0, 2, 3)
            optimizer.zero_grad()
            img_res = self.model(imgs)
            t_f = time.time()
            t_forward += t_f - t_b
            loss_2D = F.mse_loss(img_res, imgs_gt.to(img_res.device)) / self.num_cam
            loss = loss_2D
            loss.backward()
            optimizer.step()
            t_b = time.time()
            t_backward += t_b - t_f
            if (batch_idx + 1) % self.args.log_interval == 0 or batch_idx == len(data_loader) - 1 or batch_idx == 0:
                t1 = time.time()
                t_epoch = t1 - t0
                current_lr = optimizer.state_dict()['param_groups'][0]['lr']
                print(f'Train Epoch:{epoch}, Batch_id:{batch_idx + 1}, Loss:{losses / (batch_idx + 1):.6f}'
                      f'  Time:{t_epoch:.1f} f({t_forward:.1f}+b{t_backward:.1f}) lr:{current_lr:.8f}')
                # visulaization in training process
                # 2D image
                heatmap0_head = img_res[0].detach().cpu().numpy().squeeze()
                img0 = self.denormalize(imgs[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
                img0 = Image.fromarray((img0 * 255).astype('uint8'))
                head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
                head_cam_result.save(os.path.join(self.logdir, f'b_{batch_idx + 1}_cam1_head.jpg'))

            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

    def train_2D_SVP(self, data_loader, epoch, optimizer, scheduler):
        print("####################  Training 2D+SVP model  ########################")
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0
        losses = 0
        for batch_idx, (imgs, imgs_gt, view_gp_gt, gp_gt, frame) in enumerate(data_loader):
            imgs_gt = imgs_gt.permute(1, 0, 2, 3)
            view_gp_gt = view_gp_gt.permute(1, 0, 2, 3)
            optimizer.zero_grad()
            img_res, view_gp_res = self.model(imgs)
            t_f = time.time()
            t_forward += t_f - t_b
            # loss 2D
            loss_2D = F.mse_loss(img_res, imgs_gt.to(img_res.device)) / self.num_cam
            # loss SVP
            loss_svp = F.mse_loss(view_gp_res, view_gp_gt.to(view_gp_res.device)) / self.num_cam
            loss = loss_svp + loss_2D.to(loss_svp.device) * self.weight_2D

            # backward
            loss.backward()
            losses += loss.item()
            # writer.add_scalar(tag="train_loss", scalar_value=loss.item(),
            #                   global_step=epoch * len(train_loader) + batch_idx)
            optimizer.step()
            t_b = time.time()
            t_backward += t_b - t_f
            if (batch_idx + 1) % self.args.log_interval == 0 or batch_idx == len(data_loader) - 1 or batch_idx == 0:
                t1 = time.time()
                t_epoch = t1 - t0
                current_lr = optimizer.state_dict()['param_groups'][0]['lr']
                print(
                    f'Train Epoch:{epoch}, Batch_id:{batch_idx + 1}, Loss:{losses / (batch_idx + 1):.6f} [2D {loss_2D.item():.6f},'
                    f' SVP {loss_svp.item():.6f}], Time:{t_epoch:.1f} f({t_forward:.1f}+b{t_backward:.1f}) lr:{current_lr:.8f}')
                # visulaization in training process
                # 2D image
                heatmap0_head = img_res[0].detach().cpu().numpy().squeeze()
                img0 = self.denormalize(imgs[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
                img0 = Image.fromarray((img0 * 255).astype('uint8'))
                head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
                head_cam_result.save(os.path.join(self.logdir, f'b_{batch_idx + 1}_cam1_head.jpg'))
                # single-view prediction
                prediction_vis(view_gp_res[0], view_gp_gt[0],
                               save_dir=os.path.join(self.logdir, f'b_{batch_idx + 1}_cam1_gp_res.jpg'))

            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()


    def train_2D_SVP_VCW(self, data_loader, epoch, optimizer, scheduler):
        print("####################  Training 2D+SVP+VCW model  ########################")
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0
        losses = 0
        for batch_idx, (imgs, imgs_gt, view_gp_gt, gp_gt, frame) in enumerate(data_loader):
            imgs_gt = imgs_gt.permute(1, 0, 2, 3)
            view_gp_gt = view_gp_gt.permute(1, 0, 2, 3)
            optimizer.zero_grad()
            img_res, view_gp_res, gp_res = self.model(imgs)
            t_f = time.time()
            t_forward += t_f - t_b
            # loss 2D
            loss_2D = F.mse_loss(img_res, imgs_gt.to(img_res.device)) / self.num_cam
            # loss SVP
            loss_svp = F.mse_loss(view_gp_res, view_gp_gt.to(view_gp_res.device)) / self.num_cam
            # loss fusion
            loss_fusion = F.mse_loss(gp_res, gp_gt.to(gp_res.device)).to(loss_2D.device)
            loss = loss_fusion + loss_2D.to(loss_fusion.device) * self.args.weight_2D + loss_svp.to(
                loss_fusion.device) * self.args.weight_svp

            # backward
            loss.backward()
            losses += loss.item()
            optimizer.step()

            t_b = time.time()
            t_backward += t_b - t_f

            if (batch_idx + 1) % self.args.log_interval == 0 or batch_idx == len(data_loader) - 1 or batch_idx == 0:
                t1 = time.time()
                t_epoch = t1 - t0
                current_lr = optimizer.state_dict()['param_groups'][0]['lr']
                print(
                    f'Train Epoch:{epoch}, Batch_id:{batch_idx + 1}, Loss:{losses / (batch_idx + 1):.6f} [2D {loss_2D.item():.6f},'
                    f' SVP {loss_svp.item():.6f}, fusion {loss_fusion.item():.6f},], Time:{t_epoch:.1f} f({t_forward:.1f}+b{t_backward:.1f}) '
                    f'lr:{current_lr:.8f}, maxima:{gp_res.max():.3f}')
                # visulaization in training process
                # 2D image
                heatmap0_head = img_res[0].detach().cpu().numpy().squeeze()
                img0 = self.denormalize(imgs[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
                img0 = Image.fromarray((img0 * 255).astype('uint8'))
                head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
                head_cam_result.save(os.path.join(self.logdir, f'b_{batch_idx + 1}_cam1_head.jpg'))
                # single-view prediction
                prediction_vis(view_gp_res[0], view_gp_gt[0],
                               save_dir=os.path.join(self.logdir, f'b_{batch_idx + 1}_cam1_gp_res.jpg'))
                # ground plane fusion prediction
                prediction_vis(gp_res, gp_gt, save_dir=os.path.join(self.logdir, f'b_{batch_idx + 1}_gp_res.jpg'))
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

    def val(self, variant, data_loader, res_fpath, epoch):
        self.model.eval()
        losses = 0
        self.val_interval = len(data_loader) // 4
        epoch_dir = os.path.join(self.logdir, f'epoch{epoch}')
        os.makedirs(epoch_dir)
        if variant == '2D':
            self.val_2D(data_loader, losses, epoch_dir)
        elif variant == '2D_SVP':
            self.val_2D_SVP(data_loader, losses, epoch_dir)
        elif variant == '2D_SVP_VCW':
            self.val_2D_SVP_VCW(data_loader, res_fpath, losses, epoch_dir, epoch)
        else:
            raise Exception("Wrong variant.")

    def val_2D(self, data_loader, losses, epoch_dir):
        t0 = time.time()
        # data: [img_views, img_gt, camera_paras, wld_map_paras, hw_random, gp_gt, view_gp_gt]
        for batch_idx, (imgs, imgs_gt, view_gp_gt, gp_gt, frame) in enumerate(data_loader):
            imgs_gt = imgs_gt.permute(1, 0, 2, 3)
            # view_gp_gt = view_gp_gt.permute(1, 0, 2, 3)
            with torch.no_grad():
                img_res = self.model(imgs)
                loss_2D = F.mse_loss(img_res, imgs_gt.to(img_res.device)) / self.num_cam
                loss = loss_2D
                losses += loss.item()
            # visulaization in training process
            if (batch_idx + 1) % self.val_interval == 0 or batch_idx == 0:
                heatmap0_head = img_res[0].detach().cpu().numpy().squeeze()
                img0 = self.denormalize(imgs[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
                img0 = Image.fromarray((img0 * 255).astype('uint8'))
                head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
                head_cam_result.save(os.path.join(epoch_dir, f'b_{batch_idx + 1}_cam1_head.jpg'))

        t1 = time.time()
        print(f'################################### Testing #############################################\n'
              f'Batch:{len(data_loader)}, Loss:{losses / len(data_loader) :.6f}'
              f' [2D {loss_2D.item():.6f}], Time:{t1 - t0:.1f}')

    def val_2D_SVP(self, data_loader, losses, epoch_dir):
        t0 = time.time()
        for batch_idx, (imgs, imgs_gt, view_gp_gt, gp_gt, frame) in enumerate(data_loader):
            imgs_gt = imgs_gt.permute(1, 0, 2, 3)
            view_gp_gt = view_gp_gt.permute(1, 0, 2, 3)
            with torch.no_grad():
                img_res, view_gp_res = self.model(imgs)
            # loss 2D
            loss_2D = F.mse_loss(img_res, imgs_gt.to(img_res.device)) / self.num_cam
            # loss SVP
            loss_svp = F.mse_loss(view_gp_res, view_gp_gt.to(view_gp_res.device)) / self.num_cam
            loss = loss_svp + loss_2D.to(loss_svp.device) * self.args.weight_2D
            losses += loss.item()
            # visulaization in training process
            if (batch_idx + 1) % self.val_interval == 0 or batch_idx == 0:
                heatmap0_head = img_res[0].detach().cpu().numpy().squeeze()
                img0 = self.denormalize(imgs[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
                img0 = Image.fromarray((img0 * 255).astype('uint8'))
                head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
                head_cam_result.save(os.path.join(epoch_dir, f'b_{batch_idx + 1}_cam1_head.jpg'))
                # single-view prediction
                prediction_vis(view_gp_res[0, 0], view_gp_gt[0, 0, 0],
                               save_dir=os.path.join(epoch_dir, f'b_{batch_idx + 1}_cam1_gp_res.jpg'))
        t1 = time.time()
        print(f'################################### Testing #############################################\n'
              f' Batch:{len(data_loader)}, Loss:{losses / len(data_loader) :.6f}'
              f' [2D {loss_2D.item():.6f}, SVP: {loss_svp.item():.6f}], Time:{t1 - t0:.1f}')

    def val_2D_SVP_VCW(self, data_loader, res_fpath, losses, epoch_dir, epoch):
        t0 = time.time()
        all_res_list = []
        for batch_idx, (imgs, imgs_gt, view_gp_gt, gp_gt, frame) in enumerate(data_loader):
            frame = int(frame)
            imgs_gt = imgs_gt.permute(1, 0, 2, 3)
            view_gp_gt = view_gp_gt.permute(1, 0, 2, 3)
            with torch.no_grad():
                img_res, view_gp_res, gp_res = self.model(imgs)
            if res_fpath is not None:
                # get the normalized results
                map_grid_res = gp_res.detach().cpu().squeeze()
                map_grid_res = torch.relu(map_grid_res)
                map_grid_res = (map_grid_res - map_grid_res.min()) / (map_grid_res.max() - map_grid_res.min() + 1e-12)
                v_s = map_grid_res[map_grid_res > self.cls_thres].unsqueeze(1)
                grid_ij = (map_grid_res > self.cls_thres).nonzero()

                all_res_list.append(torch.cat([torch.ones_like(v_s) * frame, grid_ij.float() *
                                               data_loader.dataset.world_reduce * 4, v_s], dim=1))

                frame += 1

            # loss 2D
            loss_2D = F.mse_loss(img_res, imgs_gt.to(img_res.device)) / self.num_cam
            # loss SVP
            loss_svp = F.mse_loss(view_gp_res, view_gp_gt.to(view_gp_res.device)) / self.num_cam
            # loss fusion
            loss_fusion = F.mse_loss(gp_res, gp_gt.to(gp_res.device)).to(loss_2D.device)
            loss = loss_fusion + loss_2D.to(loss_fusion.device) * self.args.weight_2D + loss_svp.to(
                loss_fusion.device) * self.args.weight_svp
            losses += loss.item()
            if (batch_idx + 1) % self.val_interval == 0 or batch_idx == 0:
                # visulaization in training process
                # 2D image
                heatmap0_head = img_res[0].detach().cpu().numpy().squeeze()
                img0 = self.denormalize(imgs[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
                img0 = Image.fromarray((img0 * 255).astype('uint8'))
                head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
                head_cam_result.save(os.path.join(epoch_dir, f'b_{batch_idx + 1}_cam1_head.jpg'))
                # single-view prediction
                prediction_vis(view_gp_res[0], view_gp_gt[0],
                               save_dir=os.path.join(epoch_dir, f'b_{batch_idx + 1}_cam1_gp_res.jpg'))
                # ground plane fusion prediction
                prediction_vis(gp_res, gp_gt, save_dir=os.path.join(epoch_dir, f'b_{batch_idx + 1}_gp_res.jpg'))
            # if batch_idx == 5:
            #     break
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
                ids, count = nms(positions, scores, self.nms_thres, np.inf)
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            np.savetxt(res_fpath, res_list, '%d')

            recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), data_loader.dataset.gt_fpath,
                                                     self.dist_thres,
                                                     data_loader.dataset.base.__name__)
            F1_score = 2 * precision * recall / (precision + recall + 1e-12)
            print(f'moda: {moda:.1f}%, modp: {modp:.1f}%,'
                  f' precision: {precision:.1f}%, recall: {recall:.1f}%, F1_score:{F1_score:.1f}%')
            # save the best model
            if moda >= self.metrics['moda'] and F1_score >= self.metrics['f1_score']:
                checkpoint = {'model': self.model.state_dict()}
                torch.save(checkpoint, os.path.join(self.logdir, f'best_model_{epoch}.pth'))
                # delete the previous model started with 'best_model'
                for filename in os.listdir(self.logdir):
                    if filename.startswith('best_model'):
                        os.remove(os.path.join(self.logdir, filename))