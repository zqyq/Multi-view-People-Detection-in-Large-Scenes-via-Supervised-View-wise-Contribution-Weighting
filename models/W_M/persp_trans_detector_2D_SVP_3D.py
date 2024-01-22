import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from torchvision.models.vgg import vgg11
from models.resnet import resnet18
from utils.person_help import *


class PerspTransDetector_2D_SVP_3D(nn.Module):
    def __init__(self, dataset, arch='resnet18', **kwargs):
        super().__init__()
        self.num_cam = dataset.num_cam
        self.img_shape, self.reducedgrid_shape = dataset.img_shape, dataset.reducedgrid_shape
        imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(dataset.base.intrinsic_matrices,
                                                                           dataset.base.extrinsic_matrices,
                                                                           dataset.base.worldgrid2worldcoord_mat)
        self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])

        # img
        self.device = ['cuda:0', 'cuda:0']
        self.upsample_shape = list(map(lambda x: int(x / dataset.img_reduce), self.img_shape))
        img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)
        img_zoom_mat = np.diag(np.append(img_reduce, [1]))
        # map
        map_zoom_mat = np.diag(np.append(np.ones([2]) / dataset.grid_reduce, [1]))
        # projection matrices: img feat -> map feat
        self.proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
                          for cam in range(self.num_cam)]

        if arch == 'vgg11':
            base = vgg11().features
            base[-1] = nn.Sequential()
            base[-4] = nn.Sequential()
            split = 10
            self.base_pt1 = base[:split].to(self.device[0])
            self.base_pt2 = base[split:].to(self.device[0])
            out_channel = 512
        elif arch == 'resnet18':
            base = nn.Sequential(*list(resnet18(replace_stride_with_dilation=[False, True, True]).children())[:-2])
            split = 7
            self.base_pt1 = base[:split].to(self.device[0])
            self.base_pt2 = base[split:].to(self.device[0])
            out_channel = 512
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')
        # 2.5cm -> 0.5m: 20x
        self.img_classifier = nn.Sequential(nn.Conv2d(out_channel, 64, 1), nn.ReLU(),
                                            nn.Conv2d(64, 2, 1, bias=False)).to(self.device[0])

        self.view_gp_decoder = nn.Sequential(nn.Conv2d(out_channel, 512, 3, padding=1), nn.ReLU(),
                                             nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                                             nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                                             nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
                                             nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
                                             nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
                                             nn.Conv2d(64, 1, 1, bias=False)).to(self.device[1])

        # self.view_gp_decoder = nn.Sequential(nn.Conv2d(out_channel, 512, 3, padding=1), nn.ReLU(),
        #                                      # nn.Conv2d(512, 512, 5, 1, 2), nn.ReLU(),
        #                                      nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
        #                                      nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False)).to(self.device[1])

        self.view_weight = nn.Sequential(nn.Conv2d(1, 256, 3, padding=1), nn.ReLU(),
                                         nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
                                         nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
                                         nn.Conv2d(128, 1, 1, bias=False)).to(self.device[1])

        # self.view_weight=nn.Sequential(nn.Conv2d(1,64,1),nn.ReLU(),)
        self.GP_decoder = nn.Sequential(nn.Conv2d(out_channel, 512, 3, padding=1), nn.ReLU(),
                                        # nn.Conv2d(512, 512, 5, 1, 2), nn.ReLU(),
                                        nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
                                        nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False)).to(self.device[1])
        if kwargs['fix_2D'] == 1:
            print('### Fix_2D ###')
            for param_1 in self.base_pt1.parameters():
                param_1.requires_grad = False
            for param_2 in self.base_pt2.parameters():
                param_2.requires_grad = False
            for param_3 in self.img_classifier.parameters():
                param_3.requires_grad = False
        if kwargs['fix_svp'] == 1:
            print('### Fix SVP ###')
            for param_4 in self.view_gp_decoder.parameters():
                param_4.requires_grad = False
        self.Initialize(self.view_weight)
        self.Initialize(self.GP_decoder, pattern=2)

    def Initialize(self, *args, pattern=1):
        from torch.nn import init
        for modu in args:
            for layer in modu.modules():
                if isinstance(layer, nn.Conv2d):
                    if pattern == 1:
                        init.constant_(layer.weight,
                                       1 / (layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]))
                    elif pattern == 2:
                        # init.constant_(layer.weight, 0.01)
                        init.kaiming_normal_(layer.weight,nonlinearity='relu')
                    else:
                        pass
                    if layer.bias is not None:
                        init.constant_(layer.bias, 0)

    def forward(self, imgs, visualize=False):
        B, N, C, H, W = imgs.shape
        assert N == self.num_cam
        world_features = []
        img_results = []
        for cam in range(self.num_cam):
            img_feature = self.base_pt1(imgs[:, cam].to(self.device[0]))
            img_feature = self.base_pt2(img_feature.to(self.device[0]))
            img_feature = F.interpolate(img_feature, self.upsample_shape, mode='bilinear')
            img_res = self.img_classifier(img_feature.to(self.device[0]))
            img_results.append(img_res)
            proj_mat = self.proj_mats[cam].repeat([B, 1, 1]).float().to(self.device[0])
            world_feature = kornia.geometry.warp_perspective(img_feature.to(self.device[0]), proj_mat,
                                                             self.reducedgrid_shape)
            world_features.append(world_feature.to(self.device[0]))
        # world_features = torch.cat(world_features + [self.coord_map.repeat([B, 1, 1, 1]).to(self.device[0])], dim=1)
        world_features = torch.cat(world_features)
        img_results = torch.cat(img_results)
        # generate view_masks
        view_mask = (torch.max(world_features, dim=1, keepdim=True)[0] > 0).int()
        # svp
        view_gp_res = self.view_gp_decoder(world_features.to(self.device[0]))
        # gp
        weight = self.view_weight(view_gp_res)
        # weight_exp = torch.exp(weight)
        weight_mask = torch.mul(weight, view_mask)
        weight_mask_sum = torch.sum(weight_mask, dim=0, keepdim=True) + 1e-18
        weight_mask_norm = torch.div(weight_mask, weight_mask_sum)
        world_features_with_weight = torch.mul(world_features, weight_mask_norm)
        world_features_with_weight_dimsum = torch.sum(world_features_with_weight, dim=0, keepdim=True)
        map_res = self.GP_decoder(world_features_with_weight_dimsum)

        return img_results, view_mask, view_gp_res, map_res, weight

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


def test():
    from datasets.W_M.frameDataset import frameDataset
    from datasets.W_M.Wildtrack import Wildtrack
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    transform = T.Compose([T.Resize([720, 1280]),  # H,W
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), transform=transform)
    dataloader = DataLoader(dataset, 1, False, num_workers=0)
    imgs, map_gt, imgs_gt, frame = next(iter(dataloader))
    print('imgs_gt shape', imgs_gt[0].shape)
    print('map_gt shape', map_gt.shape)
    model = PerspTransDetector_2D_SVP_3D(dataset, fix_svp=1, fix_2D=1)
    # model.load_state_dict(pre['net'])
    for name, para in model.named_parameters():
        print(name,para)
    pretrained_model_dir = '/home/yunfei/Study/MVD_VCW/logs/wildtrack_frame/resnet18/2D_SVP/2023-04-06_21-55-50_(fix2D0w1)_(fixsvp0w1)_momentum0.9_' \
                           'weight_decay0.0001_lr0.1_lrsonecycle_epo200_ct0.4_nt20_dt20/MultiviewDetector.pth'
    pretrained_model = torch.load(pretrained_model_dir, map_location='cuda:0')
    model_dict = model.state_dict()
    pretrained_dict = pretrained_model['net']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    img_results, view_mask, view_gp_res, map_res, weight = model(imgs, visualize=False)
    # print('map_res shape', map_res.shape)
    # print('img_res shape', img_res[0].shape)
    # print('map_res max', map_res.max())
    # print('map_res min', map_res.min)
    pass


if __name__ == '__main__':
    test()
