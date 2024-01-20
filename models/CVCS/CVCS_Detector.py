# import numpy as np
import argparse
import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16
from multiview_detector.models.resnet import resnet18

import matplotlib.pyplot as plt
import os
from multiview_detector.models.CVCS.cvcs_proj import spatial_transoformation_layer
from multiview_detector.utils.person_help import vis
import torch.nn.functional as F


class PerspTransDetector(nn.Module):
    def __init__(self, dataset, args):  # vgg11 resnet18
        super().__init__()
        self.args = args
        self.device = self.args.devices
        self.variant = self.args.variant
        self.batch_size = dataset.batch_size
        self.view_size = dataset.view_size
        self.patch_num = dataset.patch_num
        self.cropped_size = dataset.cropped_size
        self.upsample_shape = dataset.upsample_shape
        self.num_cam = dataset.num_cam
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
            raise Exception('architecture currently support [vgg16, resnet18]')
        # 2D image decoder
        self.img_classifier = nn.Sequential(nn.Conv2d(out_channel, 64, 3, padding=1), nn.ReLU(),
                                            nn.Conv2d(64, 1, 1, bias=False)).to(self.device[0])
        # SVP(Single-view Predcition)
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
        # freeze some nets
        self.freeze()

    @staticmethod
    def freeze_net(*args):
        for net in args:
            for para in net.parameters():
                para.requires_grad = False

    # @staticmethod
    def freeze(self):
        if self.args.fix_2D == 1:
            self.freeze_net(self.base)
        if self.args.fix_svp == 1:
            self.freeze_net(self.single_view_classifier)

    def forward(self, imgs,
                camera_paras,
                wld_map_paras,
                hw_random,
                train=True):
        if self.variant == '2D':
            return self.forward_2D(imgs)
        elif self.variant == '2D_SVP':
            return self.forward_2D_SVP(imgs,
                                       camera_paras,
                                       wld_map_paras,
                                       hw_random,
                                       train=train)
        elif self.variant == '2D_SVP_VCW':
            return self.forward_2D_SVP_VCW(imgs,
                                           camera_paras,
                                           wld_map_paras,
                                           hw_random,
                                           train=train)
        else:
            raise Exception("Wrong variant.")

    def forward_2D(self, imgs, train=True):
        B, N, C, H, W = imgs.shape
        assert N == self.num_cam
        img_feature = self.base(imgs.view(-1, C, H, W).to(self.device[0]))
        img_res = self.img_classifier(img_feature)
        return img_res

    def forward_2D_SVP(self, imgs,
                       camera_paras,
                       wld_map_paras,
                       hw_random,
                       train=True,
                       visualize=False):
        B, N, C, H, W = imgs.shape
        assert N == self.num_cam
        camera_paras2_shape = (self.batch_size * self.view_size, 15)
        camera_paras2 = torch.reshape(camera_paras, shape=camera_paras2_shape)
        world_features = []
        imgs_results = []
        for cam in range(self.num_cam):
            img_feature = self.base(imgs[:, cam].to(self.device[0]))
            # img_feature = F.interpolate(img_feature, self.upsample_shape, mode='bilinear')
            img_res = self.img_classifier(img_feature)
            imgs_results.append(img_res)
            # projetion layer for CVCS dataset:
            if train:
                paras = [self.batch_size, self.view_size, self.patch_num, self.cropped_size]
            else:
                hw_random = torch.zeros(size=[1, 2, 1], dtype=torch.int32)
                paras = [self.batch_size, self.view_size, 1,
                         [int(wld_map_paras[0][3]), int(wld_map_paras[0][4])]]
            world_feature = spatial_transoformation_layer(paras,
                                                          [img_feature.to(self.device[0]),
                                                           camera_paras2[cam:cam + 1].to(self.device[0]),
                                                           wld_map_paras.to(self.device[0]),
                                                           hw_random.to(self.device[0])])
            if visualize:
                plt.imshow(torch.norm(img_feature[0].detach(), dim=0).cpu().numpy())
                plt.show()
                plt.imshow(torch.norm(world_feature[0].detach(), dim=0).cpu().numpy())
                plt.show()
            world_feature = torch.unsqueeze(world_feature, dim=1)  # [p,512,200,200] -> [p,1, 512,200,200]: [B,N,C,H,W]
            world_features.append(world_feature.to(self.device[0]))

        imgs_results = torch.cat(imgs_results)
        world_features = torch.cat(world_features, dim=1).to(self.device[1])

        view_gp_results = []
        if train:
            iter = self.patch_num
        else:
            iter = 1
        for p in range(iter):
            # single-view prediction
            view_gp_res = self.single_view_classifier(world_features[p])
            view_gp_results.append(view_gp_res)
        view_gp_results = torch.stack(view_gp_results)
        return imgs_results, view_gp_results

    def forward_2D_SVP_VCW(self,
                           imgs,
                           camera_paras,
                           wld_map_paras,
                           hw_random,
                           train=True,
                           visualize=False):

        camera_paras2_shape = (self.batch_size * self.view_size, 15)
        camera_paras2 = torch.reshape(camera_paras, shape=camera_paras2_shape)

        B, N, C, H, W = imgs.shape
        assert N == self.num_cam
        world_features = []
        imgs_results = []
        for cam in range(self.num_cam):

            img_feature = self.base(imgs[:, cam].to(self.device[0]))
            # img_feature = F.interpolate(img_feature, self.upsample_shape, mode='bilinear')
            img_res = self.img_classifier(img_feature)
            imgs_results.append(img_res)
            # projetion layer for CVCS dataset:
            if train:
                paras = [self.batch_size, self.view_size, self.patch_num, self.cropped_size]
            else:
                # cropping images from the top-left, which means getting the integral image.
                hw_random = torch.zeros((1, 2, 1), dtype=torch.int32)
                # hw_random = hw_random[:, :, 0:1]
                paras = [self.batch_size, self.view_size, 1,
                         [int(wld_map_paras[0][3]), int(wld_map_paras[0][4])]]
            world_feature = spatial_transoformation_layer(paras,
                                                          [img_feature.to(self.device[0]),
                                                           camera_paras2[cam:cam + 1].to(self.device[0]),
                                                           wld_map_paras.to(self.device[0]),
                                                           hw_random.to(self.device[0])])

            if visualize:
                plt.imshow(torch.norm(img_feature[0].detach(), dim=0).cpu().numpy())
                plt.show()
                plt.imshow(torch.norm(world_feature[0].detach(), dim=0).cpu().numpy())
                plt.show()

            world_feature = torch.unsqueeze(world_feature, dim=1)  # [p,512,200,200] -> [p,1, 512,200,200]: [B,N,C,H,W]
            world_features.append(world_feature.to(self.device[0]))

        imgs_results = torch.cat(imgs_results)
        world_features = torch.cat(world_features, dim=1).to(self.device[1])
        # world_features, max_indices = torch.max(world_features, dim=1)
        # single-view prediction
        view_gp_results = []
        gp_results = []
        # masks = []
        if train:
            iter = self.patch_num
        else:
            iter = 1
        for p in range(iter):
            view_gp_res = self.single_view_classifier(world_features[p])
            view_gp_results.append(view_gp_res)

            weight = self.GP_view_CNN(view_gp_res)
            weight = torch.exp(weight)  # To avoid the negative terms.
            mask = (torch.norm(world_features[p], dim=1, keepdim=True) > 0).int()
            # mask = (weight>self.fb_cls).int()
            # masks.append(mask)
            weight = torch.mul(weight, mask)
            weight_sum = torch.sum(weight, dim=1, keepdim=True)
            weight = torch.div(weight, weight_sum + 1e-12)
            fused_feat = torch.mul(weight, world_features[p])
            fused_feat = torch.sum(fused_feat, dim=0, keepdim=True)
            gp_res = self.map_classifier(fused_feat)
            gp_results.append(gp_res)

        view_gp_results = torch.stack(view_gp_results)
        gp_results = torch.stack(gp_results)
        # masks = torch.stack(masks)
        return imgs_results, view_gp_results, gp_results


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--variant', type=str, default='2D_SVP_VCW',
                        choices=['default', 'img_2D', '2D_SVP', '2D_VCW', '2D_SVP_VCW', 'MH_2D_SVP', 'MH_2D_SVP_VCW'])
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg16', 'resnet18'])
    parser.add_argument('-d', '--dataset', type=str, default='cvcs',
                        choices=['wildtrack', 'multiviewx', 'citystreet', 'cvcs'])
    parser.add_argument('--pretrain', type=str, default=None, help="The pretrained model which could be loaded.")
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')

    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--val_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--lrfac', type=float, default=0.01, help='generating smaller lr')
    parser.add_argument('--lr_decay', '--ld', type=float, default=1e-4)
    parser.add_argument('--lr_scheduler', '--lrs', type=str, default='onecycle', choices=['onecycle', 'lambda', 'step'])
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--update_iterations', '--ui', type=int, default=10)
    parser.add_argument('--update_lr', '--ul', type=float, default=1e-3)

    parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: None)')
    parser.add_argument('--test', type=str, default=None)

    parser.add_argument('--facofmaxgt', '--fm', type=float, default=100)
    parser.add_argument('--facofmaxgt_gp', '--fmg', type=float, default=10)

    parser.add_argument('--fix_2D', type=float, default=1)
    parser.add_argument('--fix_svp', type=float, default=0)
    parser.add_argument('--fix_weight', type=float, default=0)

    parser.add_argument('--map_sigma', '--ms', type=float, default=3)
    parser.add_argument('--img_reduce', '--ir', type=float, default=2)
    parser.add_argument('--world_reduce', '--wr', type=float, default=2)

    parser.add_argument('--weight_svp', type=float, default=10)
    parser.add_argument('--weight_2D', type=float, default=1)

    parser.add_argument('--cls_thres', '--ct', type=float, default=0.4)
    parser.add_argument('--nms_thres', '--nt', type=float, default=5)
    parser.add_argument('--dist_thres', '--dt', type=float, default=5)

    # parser.add_argument('--person_heights', '--ph', type=list, default=[0, 600, 1200, 1800])
    parser.add_argument('--person_heights', '--ph', type=str, default='1750')
    parser.add_argument('--devices', '--cd', type=list, default=['cuda:0', 'cuda:1'])
    args = parser.parse_args()
    test(args)
