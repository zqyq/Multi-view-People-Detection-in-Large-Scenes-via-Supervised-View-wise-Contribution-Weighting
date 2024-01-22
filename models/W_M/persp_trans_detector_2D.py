import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from torchvision.models.vgg import vgg11
from models.resnet import resnet18

import matplotlib.pyplot as plt


class PerspTransDetector_2D(nn.Module):
    def __init__(self, dataset, arch='resnet18'):
        super().__init__()
        self.num_cam = dataset.num_cam
        self.img_shape, self.reducedgrid_shape = dataset.img_shape, dataset.reducedgrid_shape
        # imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(dataset.base.intrinsic_matrices,
        #                                                                    dataset.base.extrinsic_matrices,
        #                                                                    dataset.base.worldgrid2worldcoord_mat)
        # self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])

        # img
        self.device = ['cuda:0', 'cuda:0']
        self.upsample_shape = list(map(lambda x: int(x / dataset.img_reduce), self.img_shape))
        img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)
        # img_zoom_mat = np.diag(np.append(img_reduce, [1]))
        # map
        # map_zoom_mat = np.diag(np.append(np.ones([2]) / dataset.grid_reduce, [1]))
        # projection matrices: img feat -> map feat
        # self.proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
        #                   for cam in range(self.num_cam)]

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
                                         nn.Conv2d(64, 1, 1, bias=False)).to(self.device[0])
    #     self.Initialize()
    #
    # def Initialize(self):
    #     from torch.nn import init
    #     for layer in self.img_classifier.modules():
    #         if isinstance(layer, nn.Conv2d):
    #             # init.constant_(layer.weight, 1 / layer.out_channels)
    #             init.kaiming_normal_(layer.weight,nonlinearity='relu')
    #             if layer.bias is not None:
    #                 init.constant_(layer.bias, 0)

    def forward(self, imgs, visualize=False):
        B, N, C, H, W = imgs.shape
        assert N == self.num_cam
        img_results = []
        for cam in range(self.num_cam):
            img_feature = self.base_pt1(imgs[:, cam].to(self.device[0]))
            img_feature = self.base_pt2(img_feature.to(self.device[0]))
            img_feature = F.interpolate(img_feature, self.upsample_shape, mode='bilinear')
            img_res = self.img_classifier(img_feature.to(self.device[0]))
            img_results.append(img_res)
            if visualize:
                vis(torch.norm(img_feature.detach(), dim=1).cpu().squeeze())
                vis(img_res.squeeze().cpu())
        img_results = torch.cat(img_results)
        return img_results


def vis(x):
    plt.imshow(x.detach().cpu().squeeze())
    plt.show()


def test():
    from datasets.W_M.frameDataset import frameDataset
    from datasets.W_M.Wildtrack import Wildtrack
    from datasets.W_M.MultiviewX import MultiviewX
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    transform = T.Compose([T.Resize([720, 1280]),  # H,W
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), transform=transform)
    dataloader = DataLoader(dataset, 1, False, num_workers=0)
    imgs, map_gt, imgs_gt, frame = next(iter(dataloader))
    model = PerspTransDetector_2D(dataset)
    img_res = model(imgs, visualize=True)
    pass


if __name__ == '__main__':
    test()
