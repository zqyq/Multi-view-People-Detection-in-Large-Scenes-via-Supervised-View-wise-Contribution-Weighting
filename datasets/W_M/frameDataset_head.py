import json
import os
import torch.nn.functional as F

import torch
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.stats import multivariate_normal
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor
import kornia

from multiview_detector.utils.projection import *
from multiview_detector.utils import gaussian_blur_detecting, gaussian_blur_counting
from multiview_detector.utils.person_help import *
import matplotlib.pyplot as plt


class frameDataset(VisionDataset):
    def __init__(self, base, train=True, _transform=ToTensor(), target_transform=ToTensor(),
                 reID=False, grid_reduce=4, img_reduce=4, train_ratio=0.9, force_download=True, **kwargs):
        # Totensor() Convert a PIL Image or numpy.ndarray to tensor. This transform does not support torchscript.
        super().__init__(base.root, transform=_transform, target_transform=target_transform)

        self.map_sigma, map_kernel_size = 5, 20
        self.img_sigma, img_kernel_size = 5, 10
        self.base = base
        self.root, self.num_cam, self.num_frame = base.root, base.num_cam, base.num_frame
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col
        self.input_img_shape = base.input_img_shape
        self.upsample_shape = list(map(lambda x: int(x / img_reduce), self.input_img_shape))
        self.reID, self.grid_reduce = reID, grid_reduce
        self.img_reduce = self.img_shape[0] // self.upsample_shape[0]
        self.reducedgrid_shape = list(map(lambda x: int(x / self.grid_reduce), self.worldgrid_shape))
        self.facofmaxgt = kwargs['facofmaxgt'] if 'facofmaxgt' in kwargs else 100
        if train:
            frame_range = range(0, int(self.num_frame * train_ratio))  # 0.9
        else:
            frame_range = range(int(self.num_frame * train_ratio), self.num_frame)

        self.img_fpaths = self.base.get_image_fpaths(frame_range)
        self.map_gt = {}
        self.imgs_head_foot_gt = {}

        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        if not os.path.exists(self.gt_fpath) or force_download:
            self.prepare_gt()
        # imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(base.intrinsic_matrices,
        #                                                                    base.extrinsic_matrices,
        #                                                                    base.worldgrid2worldcoord_mat)
        # img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)
        # img_zoom_mat = np.diag(np.append(img_reduce, [1]))
        # # map
        # map_zoom_mat = np.diag(np.append(np.ones([2]) / self.grid_reduce, [1]))
        # # projection matrices: img feat -> map feat
        # self.proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
        #                   for cam in range(self.num_cam)]
        self.download(frame_range)

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
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                i_s, j_s, v_s = [], [], []
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
                        # y_foot = min(single_pedestrian['views'][cam]['ymax'], self.img_shape[0] - 1)
                        if x > 0 and y > 0:
                            head_row_cam_s[cam].append(y_head)
                            head_col_cam_s[cam].append(x)
                            # foot_row_cam_s[cam].append(y_foot)
                            # foot_col_cam_s[cam].append(x)
                            v_cam_s[cam].append(single_pedestrian['personID'] + 1 if self.reID else 1)
                # Generate map gt
                occupancy_map = np.zeros(self.reducedgrid_shape)
                for i in range(len(v_s)):
                    cy = i_s[i]
                    cx = j_s[i]
                    center = (cx, cy)
                    occupancy_map = gaussian_blur_detecting.draw_umich_gaussian(occupancy_map, center,
                                                                                sigma=self.map_sigma)
                self.map_gt[frame] = occupancy_map

                # Generate img gt and view_gp_gt under certain cam
                self.imgs_head_foot_gt[frame] = {}
                for cam in range(self.num_cam):
                    img_gt_head = np.zeros(self.upsample_shape)
                    # pure_img_gt = img_gt_head.copy()
                    # view_world_gt = np.zeros(self.upsample_shape)
                    for i in range(len(v_cam_s[cam])):
                        cx = head_col_cam_s[cam][i] // self.img_reduce
                        cy = head_row_cam_s[cam][i] // self.img_reduce
                        center = (cx, cy)
                        # pure_img_gt[cy, cx] = 1
                        img_gt_head = gaussian_blur_counting.draw_umich_gaussian(img_gt_head, center,
                                                                                 sigma=self.map_sigma)
                    self.imgs_head_foot_gt[frame][cam] = img_gt_head
                    # proj_mat = self.proj_mats[cam].double()[None]
                    # view_world_gt = kornia.geometry.warp_perspective(torch.from_numpy(pure_img_gt)[None,None],
                    #                                                  proj_mat, (self.reducedgrid_shape[0],
                    #                                                             self.reducedgrid_shape[1]))
                    # num_positive = view_world_gt.sum()
                    # grid_xy = (view_world_gt > 0).nonzero()
                    # for i in range(num_positive.floor()):
                    #     gy = grid_xy[0]
                    #     gx = grid_xy[1]
                    #     view_world_gt = gaussian_blur_counting.draw_umich_gaussian(view_world_gt, (gx, gy),
                    #                                                                self.map_sigma)
                # img_gt_head = coo_matrix((v_cam_s[cam], (head_row_cam_s[cam], head_col_cam_s[cam])),
                #                          shape=self.img_shape)
                # img_gt_foot = coo_matrix((v_cam_s[cam], (foot_row_cam_s[cam], foot_col_cam_s[cam])),
                #                          shape=self.img_shape)

    def __getitem__(self, index):
        frame = list(self.map_gt.keys())[index]
        imgs = []
        for cam in range(self.num_cam):
            fpath = self.img_fpaths[cam][frame]
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)
        map_gt = self.map_gt[frame] * self.facofmaxgt
        # if self.reID:
        #     map_gt = (map_gt > 0).int()
        if self.target_transform is not None:
            map_gt = self.target_transform(map_gt)

        imgs_gt = []
        for cam in range(self.num_cam):
            img_gt_head = self.imgs_head_foot_gt[frame][cam] * self.facofmaxgt
            # img_gt_foot = self.imgs_head_foot_gt[frame][cam][1].toarray()
            # img_gt = np.stack([img_gt_head, img_gt_foot], axis=2)
            # if self.reID:
            #     img_gt_head = (img_gt_head > 0).int()
            if self.target_transform is not None:
                img_gt_head = self.target_transform(img_gt_head)
            imgs_gt.append(img_gt_head.float())
        imgs_gt = torch.stack(imgs_gt)
        return imgs, map_gt.float(), imgs_gt, frame

    def __len__(self):
        return len(self.map_gt.keys())


def test():
    from multiview_detector.datasets.W_M.Wildtrack import Wildtrack
    # from multiview_detector.datasets.MultiviewX import MultiviewX
    from multiview_detector.utils.projection import get_worldcoord_from_imagecoord
    dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')))
    # test projection
    # world_grid_maps = []
    # xx, yy = np.meshgrid(np.arange(0, 1920, 20), np.arange(0, 1080, 20))
    # H, W = xx.shape
    # image_coords = np.stack([xx, yy], axis=2).reshape([-1, 2])
    # import matplotlib.pyplot as plt
    # for cam in range(dataset.num_cam):
    #     world_coords = get_worldcoord_from_imagecoord(image_coords.transpose(), dataset.base.intrinsic_matrices[cam],
    #                                                   dataset.base.extrinsic_matrices[cam])
    #     world_grids = dataset.base.get_worldgrid_from_worldcoord(world_coords).transpose().reshape([H, W, 2])
    #     world_grid_map = np.zeros(dataset.worldgrid_shape)
    #     for i in range(H):
    #         for j in range(W):
    #             x, y = world_grids[i, j]
    #             if dataset.base.indexing == 'xy':
    #                 if x in range(dataset.worldgrid_shape[1]) and y in range(dataset.worldgrid_shape[0]):
    #                     world_grid_map[int(y), int(x)] += 1
    #             else:
    #                 if x in range(dataset.worldgrid_shape[0]) and y in range(dataset.worldgrid_shape[1]):
    #                     world_grid_map[int(x), int(y)] += 1
    #     world_grid_map = world_grid_map != 0
    #     plt.imshow(world_grid_map)
    #     plt.show()
    #     world_grid_maps.append(world_grid_map)
    #     pass
    # plt.imshow(np.sum(np.stack(world_grid_maps), axis=0))
    # plt.show()
    pass
    imgs, map_gt, imgs_gt, _ = dataset.__getitem__(0)

    pass


# def vis(x):
#     plt.imshow(x.detach().cpu().squeeze())
#     plt.show()


# def trans(x, target, kernel):
#     target = F.adaptive_max_pool2d(target, x.shape[2:])
#     with torch.no_grad():
#         target = F.conv2d(target, kernel.float().to(target.device), padding=int((kernel.shape[-1] - 1) / 2))
#     return target


if __name__ == '__main__':
    test()
