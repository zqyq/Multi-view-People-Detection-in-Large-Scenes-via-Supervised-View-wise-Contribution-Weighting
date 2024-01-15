import datetime
import json
import os
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from scipy.sparse import coo_matrix
from scipy.stats import multivariate_normal
from torch import nn
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor

from multiview_detector.utils.projection import *


class frameDataset(VisionDataset):
    def __init__(self, base, train=True, transform=ToTensor(), target_transform=ToTensor(),
                 reID=False, grid_reduce=4, img_reduce=4, train_ratio=0.9, force_download=True):
        # Totensor() Convert a PIL Image or numpy.ndarray to tensor. This transform does not support torchscript.
        super().__init__(base.root, transform=transform, target_transform=target_transform)

        map_sigma, map_kernel_size = 20 / grid_reduce, 20
        img_sigma, img_kernel_size = 20 / img_reduce, 20
        self.reID, self.grid_reduce, self.img_reduce = reID, grid_reduce, img_reduce

        self.base = base
        self.root, self.num_cam, self.num_frame = base.root, base.num_cam, base.num_frame
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col
        self.reducedgrid_shape = list(map(lambda x: int(x / self.grid_reduce), self.worldgrid_shape))

        if train:
            frame_range = range(0, int(self.num_frame * train_ratio))  # 0.9
        else:
            frame_range = range(int(self.num_frame * train_ratio), self.num_frame)

        self.img_fpaths = self.base.get_image_fpaths(frame_range)
        self.map_gt = {}
        self.imgs_head_foot_gt = {}
        self.download(frame_range)

        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        if not os.path.exists(self.gt_fpath) or force_download:
            self.prepare_gt()

        x, y = np.meshgrid(np.arange(-map_kernel_size, map_kernel_size + 1),
                           np.arange(-map_kernel_size, map_kernel_size + 1))

        pos = np.stack([x, y], axis=2)
        map_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * map_sigma)
        map_kernel = map_kernel / map_kernel.max()
        kernel_size = map_kernel.shape[0]
        self.map_kernel = torch.zeros([1, 1, kernel_size, kernel_size], requires_grad=False)
        self.map_kernel[0, 0] = torch.from_numpy(map_kernel)

        x, y = np.meshgrid(np.arange(-img_kernel_size, img_kernel_size + 1),
                           np.arange(-img_kernel_size, img_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        img_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * img_sigma)
        img_kernel = img_kernel / img_kernel.max()
        kernel_size = img_kernel.shape[0]
        self.img_kernel = torch.zeros([2, 2, kernel_size, kernel_size], requires_grad=False)
        self.img_kernel[0, 0] = torch.from_numpy(img_kernel)
        self.img_kernel[1, 1] = torch.from_numpy(img_kernel)
        pass

    def prepare_gt(self):
        og_gt = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)
            for single_pedestrian in all_pedestrians:
                def is_in_cam(cam):
                    return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                single_pedestrian['views'][cam]['xmax'] == -1 and
                                single_pedestrian['views'][cam]['ymin'] == -1 and
                                single_pedestrian['views'][cam]['ymax'] == -1)

                in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                if not in_cam_range:
                    continue
                grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                og_gt.append(np.array([frame, grid_x, grid_y]))

        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def download(self, frame_range):
        # 每张图片中的每个人在每个相机下
        # 第一个for循环生成map_gt列表，包含1800帧图像，一个元素代表对应图像中每个人的位置
        # 第二个for循环生成imgs_head_foot列表，一个元素代表一帧图像中所有人在所有相机视角下head和foot位置
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                i_s, j_s, v_s = [], [], []  # world grid coord
                head_row_cam_s, head_col_cam_s = [[] for _ in range(self.num_cam)], \
                                                 [[] for _ in range(self.num_cam)]
                foot_row_cam_s, foot_col_cam_s, v_cam_s = [[] for _ in range(self.num_cam)], \
                                                          [[] for _ in range(self.num_cam)], \
                                                          [[] for _ in range(self.num_cam)]

                for single_pedestrian in all_pedestrians:
                    x, y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                    if self.base.indexing == 'xy':
                        i_s.append(int(y / self.grid_reduce))
                        j_s.append(int(x / self.grid_reduce))
                    else:
                        i_s.append(int(x / self.grid_reduce))
                        j_s.append(int(y / self.grid_reduce))
                    v_s.append(single_pedestrian['personID'] + 1 if self.reID else 1)
                    for cam in range(self.num_cam):
                        x = max(min(int((single_pedestrian['views'][cam]['xmin'] +
                                         single_pedestrian['views'][cam]['xmax']) / 2), self.img_shape[1] - 1), 0)
                        y_head = max(single_pedestrian['views'][cam]['ymin'], 0)
                        y_foot = min(single_pedestrian['views'][cam]['ymax'], self.img_shape[0] - 1)
                        if x > 0 and y > 0:
                            head_row_cam_s[cam].append(y_head)
                            head_col_cam_s[cam].append(x)
                            foot_row_cam_s[cam].append(y_foot)
                            foot_col_cam_s[cam].append(x)
                            v_cam_s[cam].append(single_pedestrian['personID'] + 1 if self.reID else 1)
                occupancy_map = coo_matrix((v_s, (i_s, j_s)), shape=self.reducedgrid_shape)  # 120, 360
                self.map_gt[frame] = occupancy_map
                self.imgs_head_foot_gt[frame] = {}
                for cam in range(self.num_cam):
                    img_gt_head = coo_matrix((v_cam_s[cam], (head_row_cam_s[cam], head_col_cam_s[cam])),
                                             shape=self.img_shape)
                    img_gt_foot = coo_matrix((v_cam_s[cam], (foot_row_cam_s[cam], foot_col_cam_s[cam])),
                                             shape=self.img_shape)
                    self.imgs_head_foot_gt[frame][cam] = [img_gt_head, img_gt_foot]

    def __getitem__(self, index):  ####
        frame = list(self.map_gt.keys())[index]
        imgs = []
        for cam in range(self.num_cam):
            fpath = self.img_fpaths[cam][frame]
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)
        map_gt = self.map_gt[frame].toarray()
        if self.reID:
            map_gt = (map_gt > 0).int()
        if self.target_transform is not None:
            map_gt = self.target_transform(map_gt)
        imgs_gt = []
        for cam in range(self.num_cam):
            img_gt_head = self.imgs_head_foot_gt[frame][cam][0].toarray()
            img_gt_foot = self.imgs_head_foot_gt[frame][cam][1].toarray()
            img_gt = np.stack([img_gt_head, img_gt_foot], axis=2)
            if self.reID:
                img_gt = (img_gt > 0).int()
            if self.target_transform is not None:
                img_gt = self.target_transform(img_gt)
            imgs_gt.append(img_gt.float())
        imgs_gt = torch.stack(imgs_gt)
        return imgs, map_gt.float(), imgs_gt, frame

    def __len__(self):
        return len(self.map_gt.keys())


def test():
    from datasets.W_M.Wildtrack import Wildtrack
    # from datasets.MultiviewX import MultiviewX
    from multiview_detector.utils.projection import get_worldcoord_from_imagecoord
    dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')))
    import torch.nn.functional as F
    imgs, map_gt, imgs_gt, _ = dataset.__getitem__(0)

    print('map_gt shape', map_gt.shape)
    print('img_gt shape', imgs_gt[0].shape)
    print('sum of imgs-gt[0]', imgs_gt[0].sum().item())
    print('img max', imgs_gt[0].max())
    print('kernel shape==', dataset.map_kernel.shape)

    map_gt = F.conv2d(map_gt.detach().unsqueeze(0), dataset.map_kernel.float(),
                      padding=int((dataset.map_kernel.shape[-1] - 1) / 2))
    img_gt0 = F.adaptive_max_pool2d(imgs_gt[0][None], (270, 480))
    img_gt0 = F.conv2d(img_gt0, dataset.img_kernel.float(),
                       padding=int((dataset.img_kernel.shape[-1] - 1) / 2))
    print('img_gt0 shape', img_gt0.shape)
    # plt.imshow(map_gt.squeeze())
    # plt.show()
    plt.imshow(img_gt0[0][0].squeeze())
    plt.show()
    plt.imshow(img_gt0[0][1].squeeze())
    plt.show()
    print(datetime.datetime.now())


if __name__ == '__main__':
    test()
