# import numpy as np
import argparse
import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16
from multiview_detector.models.resnet import resnet18

import matplotlib.pyplot as plt
import os
from multiview_detector.models.CityStreet.spatial_transformer import SpatialTransformer_v3
from multiview_detector.utils.person_help import vis, savefeatvis
import torch.nn.functional as F


class PerspTransDetector(nn.Module):
    def __init__(self, dataset, args, logdir):  # vgg11 resnet18
        super().__init__()
        self.args = args
        self.arch = self.args.arch
        self.device = self.args.devices
        self.variant = self.args.variant
        self.num_cam = dataset.num_cam
        self.hfwf = dataset.hfwf  # shape of img feature
        self.hgwg = dataset.hgwg  # shape of grid on the ground

        self.logdir = logdir
        if self.arch == 'vgg16':
            base = vgg16().features
            split = 16  # 7 conv layers, and 2 maxpooling layers.
            self.base = base[:split].to(self.device[0])
            out_channel = 256
        elif self.arch == 'resnet18':
            self.base = nn.Sequential(*list(resnet18(pretrained=True,
                                                     replace_stride_with_dilation=[False, True, True]).children())[
                                       :-2]).to(self.devic[0])
            out_channel = 512
        else:
            raise Exception('architecture currently support [vgg16, resnet18]')
        # 2D image decoder
        # self.img_classifier = nn.Sequential(nn.Conv2d(out_channel, 64, 3, padding=1), nn.ReLU(),
        #                                     nn.Conv2d(64, 1, 1, bias=False)).to(self.device[0])
        self.img_classifier = nn.Sequential(
            nn.Conv2d(out_channel, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
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
        # self.map_classifier = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
        #                                     # # w/o large kernel
        #                                     # nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
        #                                     # nn.Conv2d(512, 1, 3, padding=1, bias=False)).to(self.device[1])
        #
        #                                     # with large kernel
        #                                     nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
        #                                     nn.Conv2d(512, 1, 3, padding=1, bias=False)).to(self.device[1])
        self.map_classifier = nn.Sequential(nn.Conv2d(out_channel, 512, 3, padding=1), nn.ReLU(),
                                            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                                            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                                            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
                                            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
                                            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
                                            nn.Conv2d(64, 1, 1, bias=False)).to(self.device[1])
        # weight net
        self.GP_view_CNN = nn.Sequential(nn.Conv2d(1, 256, 3, padding=1), nn.ReLU(),
                                         nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
                                         nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
                                         nn.Conv2d(128, 1, 1, bias=False)).to(self.device[1])

        # freeze some nets
        self.freeze()

        self.STN = SpatialTransformer_v3(input_size=[1, *self.hfwf, 1], output_size=self.hgwg,
                                         device=self.device[1],
                                         person_heights=args.person_heights)

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

    def forward(self, imgs, visualize=False):
        B, N, C, H, W = imgs.shape
        assert N == 3
        imgs = imgs.view(-1, C, H, W).to(self.device[0])
        img_feature = self.base(imgs)
        img_feature = F.interpolate(img_feature, self.hfwf, mode='bilinear')
        # img_2D img_decoding
        img_res = self.img_classifier(img_feature)
        if self.variant == '2D':
            return img_res

        world_feature = self.STN(img_feature.permute(0, 2, 3, 1), self.args.person_heights[0]).permute(0, 3, 1, 2)
        view_gp_res = self.single_view_classifier(world_feature)
        if self.variant == '2D_SVP':
            return img_res, view_gp_res

        weight = self.GP_view_CNN(view_gp_res)
        # weight = torch.exp(weight)
        mask = (torch.norm(world_feature, dim=1, keepdim=True) > 0).int()
        weight = torch.mul(weight, mask)
        weight_sum = torch.sum(weight, dim=1, keepdim=True)
        weight = torch.div(weight, weight_sum + 1e-12)
        if visualize:
            for view in range(self.num_cam):
                savefeatvis(weight[view:view + 1], save=os.path.join(self.logdir, f'view{view + 1}_weight.jpg'))
        fused_feat = torch.mul(weight, world_feature)
        fused_feat = torch.sum(fused_feat, dim=0, keepdim=True)
        gp_res = self.map_classifier(fused_feat)
        return img_res, view_gp_res, gp_res


def test(args):
    # from utils.person_help import vis
    from multiview_detector.utils.load_model import loadModel
    import torchvision.transforms as transforms
    from multiview_detector.datasets.CityStreet.Citystreet import Citystreet
    from multiview_detector.datasets.CityStreet.frame_CityStreet import frameDataset
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from torch.utils.data import DataLoader
    _transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    base = Citystreet(root='/mnt/data/Datasets/CityStreet')
    train_set = frameDataset(base, facofmaxgt=100, facofmaxgt_gp=10)
    model = PerspTransDetector(train_set, args).cuda()
    pretrained_model_dir = '/mnt/data/Yunfei/Study/Baseline_MVDet/logs/citystreet_frame' \
                           '/2D_SVP_3D/2022-11-17_09-56-39_iccv/MultiviewDetector_100.pth'
    model = loadModel(model, pretrained_model_dir)

    imgs, imgs_gt, view_gp_gt, gp_gt, frame = train_set.__getitem__(0)
    img_res, view_gp_res, gp_res = model(imgs)
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
