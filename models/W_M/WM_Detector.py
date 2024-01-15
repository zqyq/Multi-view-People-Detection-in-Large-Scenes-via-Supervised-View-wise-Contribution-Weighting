import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from torchvision.models.vgg import vgg11, vgg16
from multiview_detector.models.resnet import resnet18
from multiview_detector.utils.person_help import *
import matplotlib.pyplot as plt
from multiview_detector.utils.person_help import vis


class PerspTransDetector(nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        self.args = args
        self.num_cam = dataset.num_cam
        self.img_shape, self.reducedgrid_shape = dataset.img_shape, dataset.reducedgrid_shape
        imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(dataset.base.intrinsic_matrices,
                                                                           dataset.base.extrinsic_matrices,
                                                                           dataset.base.worldgrid2worldcoord_mat)
        self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])

        # img
        self.device = [self.args.devices[0], self.args.devices[0]]  # for test, only one gpu is enough.
        self.upsample_shape = list(map(lambda x: int(x / dataset.img_reduce), self.img_shape))
        img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)
        img_zoom_mat = np.diag(np.append(img_reduce, [1]))
        # map
        map_zoom_mat = np.diag(np.append(np.ones([2]) / dataset.grid_reduce, [1]))
        # projection matrices: img feat -> map feat
        self.proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
                          for cam in range(self.num_cam)]

        if self.args.arch == 'vgg16':
            base = vgg16().features
            split = 16  # 7 conv layers, and 2 maxpooling layers.
            self.base_pt = base[:split].to(self.device[0])
            out_channel = 256
        elif self.args.arch == 'resnet18':
            self.base = nn.Sequential(*list(resnet18(pretrained=True,
                                                     replace_stride_with_dilation=[False, True, True]).children())[
                                       :-2]).to(self.device[0])
            out_channel = 512
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')
        # 2.5cm -> 0.5m: 20x
        self.img_classifier = nn.Sequential(nn.Conv2d(out_channel, 64, 3, padding=1), nn.ReLU(),
                                            nn.Conv2d(64, 1, 1, bias=False)).to(self.device[0])

        self.single_view_classifier = nn.Sequential(nn.Conv2d(out_channel, 512, 3, padding=1), nn.ReLU(),
                                                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                                                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                                                    nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
                                                    nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
                                                    nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
                                                    nn.Conv2d(64, 1, 1, bias=False)).to(self.device[1])

        # GP decoder (ground plane decoder)
        self.map_classifier = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                                            # # w/o large kernel
                                            # nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                                            # nn.Conv2d(512, 1, 3, padding=1, bias=False)).to(self.device[1])

                                            # with large kernel
                                            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                                            nn.Conv2d(512, 1, 3, padding=1, bias=False)).to(self.device[1])
        # weight net
        self.GP_view_CNN = nn.Sequential(nn.Conv2d(1, 256, 3, padding=1), nn.ReLU(),
                                         nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
                                         nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
                                         nn.Conv2d(128, 1, 1, bias=False)).to(self.device[1])

    def forward(self, imgs, visualize=False):
        B, N, C, H, W = imgs.shape
        assert N == self.num_cam
        world_features = []
        img_results = []
        for cam in range(self.num_cam):
            img_feature = self.base(imgs[:, cam].to(self.device[0]))
            img_feature = F.interpolate(img_feature, self.upsample_shape, mode='bilinear')
            img_res = self.img_classifier(img_feature.to(self.device[0]))
            img_results.append(img_res)
            proj_mat = self.proj_mats[cam].repeat([B, 1, 1]).float().to(self.device[0])
            world_feature = kornia.geometry.warp_perspective(img_feature.to(self.device[0]), proj_mat,
                                                             self.reducedgrid_shape)
            world_features.append(world_feature)
        # world_features = torch.cat(world_features + [self.coord_map.repeat([B, 1, 1, 1]).to(self.device[0])], dim=1)
        world_features = torch.cat(world_features).to(self.device[1])
        img_results = torch.cat(img_results)
        # svp
        view_gp_res = self.single_view_classifier(world_features)
        # gp
        weight = self.GP_view_CNN(view_gp_res)
        weight = torch.exp(weight)  # To avoid the negative terms.
        mask = (torch.norm(world_features, dim=1, keepdim=True) > 0).int()
        weight = torch.mul(weight, mask)
        weight_sum = torch.sum(weight, dim=1, keepdim=True)
        weight = torch.div(weight, weight_sum + 1e-12)
        fused_feat = torch.mul(weight, world_features)
        fused_feat = torch.sum(fused_feat, dim=0, keepdim=True)
        gp_res = self.map_classifier(fused_feat)

        return img_results, view_gp_res, gp_res, mask

    def get_imgcoord2worldgrid_matrices(self, intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
        projection_matrices = {}
        for cam in range(self.num_cam):
            worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)
            worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
            imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
            # image of shape C,H,W (C,N_row,N_col); indexed as x,y,w,h (x,y,n_col,n_row)
            # matrix of shape N_row, N_col; indexed as x,y,n_row,n_col
            permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
            # projection_matrices[cam]=imgcoord2worldgrid_mat
            pass
        return projection_matrices

    def create_coord_map(self, img_size, with_r=False):
        H, W, C = img_size
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
        grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
        ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        if with_r:
            rr = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
            ret = torch.cat([ret, rr], dim=1)
        return ret


def test(args):
    # from utils.person_help import vis
    from multiview_detector.utils.load_model import loadModel
    import torchvision.transforms as transforms
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    from multiview_detector.datasets.CVCS.CVCS import CVCS
    from multiview_detector.datasets.CVCS.frame_CVCS import frameDataset
    from torch.utils.data import DataLoader
    _transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    base = CVCS(root='/mnt/data/Datasets/CVCS')
    train_set = frameDataset(base)
    # train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    # img_views, imgs_gt, camera_paras, wld_map_paras, hw_random, GP_density_map = train_set.__getitem__(0)
    # img_views, imgs_gt, camera_paras, wld_map_paras, hw_random, GP_density_map = next(iter(train_loader))
    # val_set = frameDataset(base, train=False, _transform=_transform)
    # val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)
    # img_views, single_view_dmaps, camera_paras, wld_map_paras, hw_random, GP_density_map = next(iter(val_loader))
    model = PerspTransDetector(train_set, args).cuda()
    pretrained_model_dir = '/mnt/data/Yunfei/Domain-Adaptation/yunfei_model/checkpoints/MultiviewDetector_epoch50.pth'
    model = loadModel(model, pretrained_model_dir)
    # gp_res, view_gp_res, img_res, mask = model(img_views, camera_paras, wld_map_paras, hw_random, train=False)
    # print(f'gp_res:{gp_res.shape}, view_gp_res:{view_gp_res.shape}, img_res:{img_res.shape}')
    pass
