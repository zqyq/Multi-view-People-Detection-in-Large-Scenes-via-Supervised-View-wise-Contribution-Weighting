import os
# import random
import numpy as np
import torch
from PIL import Image
import h5py
from matplotlib import pyplot as plt
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor

# from datasets.CityStreet.datagen import conv_process
from datasets.CityStreet.view_mask import get_view_gp_mask
from utils.image_utils import img_color_denormalize
from utils.gaussian_blur_detecting import draw_umich_gaussian
from utils.person_help import vis


# from utils.gaussian_blur_counting import draw_umich_gaussian, gaussian2D

class frameDataset(VisionDataset):
    def __init__(self, args, base, train=True, _transform=None):
        # Totensor() Convert a PIL Image or numpy.ndarray to tensor. This transform does not support torchscript.
        super().__init__(base.root)
        self.args = args
        self.map_sigma = args.map_sigma
        self.img_shape = base.img_shape
        self.train = train
        self.base = base
        self.root, self.num_frame = base.root, base.num_frame
        self.ground_plane_shape = base.worldgrid_shape
        self.hfwf = (380, 676)
        self.img_reduce = args.img_reduce
        self.world_reduce = args.world_reduce
        self.hgwg = tuple(map(lambda x: int(x / self.world_reduce), self.ground_plane_shape))
        self.num_cam = base.num_cam
        self.facofmaxgt = args.facofmaxgt
        self.facofmaxgt_gp = args.facofmaxgt_gp

        view_masks = []
        for view in range(1, self.num_cam + 1):
            view_masks.append(get_view_gp_mask(base.root, view, self.hgwg))
        self.view_masks = np.stack(view_masks)

        # self.depth_map = self.get_depth_maps()

        self.transform = _transform
        frame_rangelist = [range(636, 1236, 2), range(1236, 1636, 2)]
        if self.train:
            self.frame_range = frame_rangelist[0]
        else:
            self.frame_range = frame_rangelist[1]
        self.img_fpaths = self.base.get_img_fpath()
        self.map_gt_from_coords = {}
        self.view_gp_gt = {_: [] for _ in self.frame_range}
        self.map_gt_from_density_maps = {}
        self.imgs_head_gt = {view: {} for view in range(1, 4)}

        self.download(self.frame_range)  # 获得map_gt,和 imgs_gt

        self.gt_fpath = os.path.join(self.root, 'gt_pixel_2.5cm.txt')
        if not os.path.exists(self.gt_fpath):
            self.prepare_gt()

    # 得到每个image中所有人的map_gt[frame],imgs_head_foot[frame]，city数据集下只考虑它的头所在位置，
    def get_dmaps_path(self):
        aimdir = {
            'gp_train': os.path.join(self.root, 'GT_density_maps/'
                                                'ground_plane/train/Street_groundplane_train_dmaps_10.h5'),
            'gp_test': os.path.join(self.root, 'GT_density_maps/'
                                               'ground_plane/test/Street_groundplane_test_dmaps_10.h5'),
            'v1_train': os.path.join(self.root, 'GT_density_maps/camera_view/train/Street_view1_dmap_10.h5'),
            'v2_train': os.path.join(self.root, 'GT_density_maps/camera_view/train/Street_view2_dmap_10.h5'),
            'v3_train': os.path.join(self.root, 'GT_density_maps/camera_view/train/Street_view3_dmap_10.h5'),

            'v1_test': os.path.join(self.root, 'GT_density_maps/camera_view/test/Street_view1_dmap_10.h5'),
            'v2_test': os.path.join(self.root, 'GT_density_maps/camera_view/test/Street_view2_dmap_10.h5'),
            'v3_test': os.path.join(self.root, 'GT_density_maps/camera_view/test/Street_view3_dmap_10.h5')}
        return aimdir

    def load_h5(self, h5dir, frame_range_list):
        temp_gt = {}
        with h5py.File(h5dir, 'r') as fp:
            dmap_i = fp['density_maps']
            dmap_i = np.squeeze(dmap_i).astype(np.float32)
            # print('dmap_i shape', dmap_i.shape)
            for i in range(0, dmap_i.shape[0]):
                temp_gt[frame_range_list[i]] = dmap_i[i][:][:]
        return temp_gt

    # 因为每帧图像对应的裁剪位置不一样，则有不同的gt坐标，所以这个总的gt文件要在trainer.py中完成
    # gp_map 就是对(os.path.expanduser('~/Data/CityStreet/Street_groundplane_pmap.h5')提取。
    def prepare_gt(self):
        og_gt = []
        with h5py.File('/mnt/data/Datasets/CityStreet/Street_groundplane_pmap.h5', 'r') as f:
            for i in range(f['v_pmap_GP'].shape[0]):
                singlePerson_Underframe = f['v_pmap_GP'][i]
                frame = int(singlePerson_Underframe[0])
                # personID = int(singlePerson_Underframe[1])
                # grid_y 为W这条边，singleFrame_underframe[2]指的是cx，最大值不超过640
                # grid_x 为H这条边，singleFrame_underframe[3]指的是cy，最大值不超过768
                grid_y = int(singlePerson_Underframe[2] * 4)  # 现在每个pixel代表2.5cm
                grid_x = int(singlePerson_Underframe[3] * 4)  # [768，640]*4
                # height = int(singlePerson_Underframe[4])
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    # og_gt = np.stack(og_gt, axis=0)
    # os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
    # np.savetxt(self.gt_fpath, og_gt, '%d')
    # def get_depth_maps(self):
    #     depth_maps_dir = '/home/yunfei/Data/CityStreet/ROI_maps/Distance_maps'
    #     depth_map_dir_list = sorted(os.listdir(depth_maps_dir))
    #     depth_map_array_list = []
    #     for i_dir in depth_map_dir_list:
    #         a = np.load(os.path.join(depth_maps_dir, i_dir))['arr_0']  # [380,676]
    #         depth_map_array_list.append(a)
    #     depth_map = np.stack(depth_map_array_list, axis=0)
    #     depth_map = torch.from_numpy(depth_map)
    #     # depth_map = torch.pow(depth_map, 0.25)
    #     # depth_map /= 1000
    #     # depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) + 1e-18
    #     # depth_map = -torch.log(depth_map)
    #     # depth_map = 1 - depth_map
    #     depth_map = depth_map.min() / depth_map
    #     return depth_map
    def download(self, frame_range):
        aimdir = self.get_dmaps_path()
        # map_gt [768,640]
        # if self.mapgtfromcoord is True:
        with h5py.File(os.path.join(self.root, 'Street_groundplane_pmap.h5'), 'r') as f:
            grounddata = f['v_pmap_GP']
            zero_array = np.zeros(self.hgwg)
            for frame in frame_range:
                occupancy_info = (grounddata[grounddata[:, 0] == frame, 2:4])
                occupancy_map = np.zeros(self.hgwg)
                view_masked_occupancy_maps = np.zeros((self.num_cam, *self.hgwg))
                for idx in range(occupancy_info.shape[0]):
                    cx, cy = occupancy_info[idx]
                    cx = int(cx // self.world_reduce)
                    cy = int(cy // self.world_reduce)
                    center = (cx, cy)
                    # create ground truth
                    occupancy_map = draw_umich_gaussian(occupancy_map, center, sigma=self.map_sigma)
                    # judge whether the point 'center' is within certain view-area on ground plane.
                    zero_array[cy, cx] = 10
                    for view in range(self.num_cam):
                        if (zero_array * self.view_masks[view]).max() == 10:
                            view_masked_occupancy_maps[view] = draw_umich_gaussian(view_masked_occupancy_maps[view],
                                                                                   center, self.map_sigma)
                    zero_array[cy, cx] = 0

                self.map_gt_from_coords[frame] = occupancy_map
                # view_gp_gt
                self.view_gp_gt[frame] = torch.from_numpy(view_masked_occupancy_maps)

        # imgs_gt [380,676]
        if self.train:
            # temp_gp_train = self.load_h5(aimdir['gp_train'], frame_range)
            temp_view1_train = self.load_h5(aimdir['v1_train'], frame_range)
            temp_view2_train = self.load_h5(aimdir['v2_train'], frame_range)
            temp_view3_train = self.load_h5(aimdir['v3_train'], frame_range)
            for i in frame_range:
                # temp_gp_train[i] = conv_process(temp_gp_train[i][:, :, None], stride=self.world_reduce,
                #                                 filter_size=self.world_reduce)
                # self.map_gt_from_density_maps[i] = temp_gp_train[i]
                self.imgs_head_gt[1][i] = temp_view1_train[i]
                self.imgs_head_gt[2][i] = temp_view2_train[i]
                self.imgs_head_gt[3][i] = temp_view3_train[i]
        else:
            # temp_gp_test = self.load_h5(aimdir['gp_test'], frame_range)
            temp_view1_test = self.load_h5(aimdir['v1_test'], frame_range)
            temp_view2_test = self.load_h5(aimdir['v2_test'], frame_range)
            temp_view3_test = self.load_h5(aimdir['v3_test'], frame_range)
            for i in frame_range:
                # temp_gp_test[i] = conv_process(temp_gp_test[i][:, :, None], stride=self.world_reduce,
                #                                filter_size=self.world_reduce)
                # self.map_gt_from_density_maps[i] = temp_gp_test[i]
                self.imgs_head_gt[1][i] = temp_view1_test[i]
                self.imgs_head_gt[2][i] = temp_view2_test[i]
                self.imgs_head_gt[3][i] = temp_view3_test[i]

    def __getitem__(self, index):
        frame = self.frame_range[index]
        imgs = []
        for view in range(1, 4):
            fpath = self.img_fpaths[view][frame]
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)

        # img_gt
        imgs_gt = []
        for view in range(1, 4):
            img_gt = self.imgs_head_gt[view][frame] * self.facofmaxgt
            imgs_gt.append(img_gt[None])
        imgs_gt = torch.from_numpy(np.concatenate(imgs_gt))

        # ground plane gt
        gp_gt = torch.from_numpy(self.map_gt_from_coords[frame])[None]

        # masked view_gp_gt on the ground plane
        view_gp_gt = self.view_gp_gt[frame]

        return imgs, imgs_gt.float(), view_gp_gt.float(), gp_gt.float(), frame

    def __len__(self):
        return len(self.frame_range)


def test(args):
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    img_reduce = 2
    world_reduce = 2
    train_trans = T.Compose([T.Resize([1520 // img_reduce, 2704 // img_reduce]), T.ToTensor(), normalize])
    trainset = frameDataset(args, Citystreet(os.path.expanduser('/mnt/data/Datasets/CityStreet')),
                            _transform=train_trans)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    imgs, imgs_gt, view_gp_gt, gp_gt, frame = next(iter(trainloader))
    pass


if __name__ == '__main__':
    import argparse
    from datasets.CityStreet.Citystreet import Citystreet
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

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
    parser.add_argument('--facofmaxgt_gp', '--fmg', type=float, default=1)

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
    parser.add_argument('--devices', '--cd', type=list, default=['cuda:2', 'cuda:3'])
    args = parser.parse_args()
    test(args)
