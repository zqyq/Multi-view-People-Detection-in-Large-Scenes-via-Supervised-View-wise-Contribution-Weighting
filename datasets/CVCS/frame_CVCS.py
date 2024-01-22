import matplotlib.pyplot as plt
import torch

from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
from utils.image_utils import draw_umich_gaussian, gaussian2D
import os
# import sys
import numpy as np
import json
import cv2
import random
from PIL import Image
from utils.person_help import vis
import scipy


class frameDataset(VisionDataset):
    def __init__(self, base, train=True, _transform=None, variant='2D', img_sigma=3, map_sigma=3, visualize=False):
        super(frameDataset, self).__init__(base)
        self.__name__ = base.__name__
        self.base = base
        self.variant = variant
        self.upsample_shape = [1080 // 8, 1920 // 8]
        self.img_sigma = img_sigma
        self.map_sigma = map_sigma
        self.visualize = visualize
        # self.gaussian_kernel_sum = gaussian2D((6 * img_sigma + 1, 6 * img_sigma + 1), sigma=img_sigma).sum()
        self.facofmaxgt = 100
        self.transform = _transform
        # read images
        self.train = train

        if train:
            self.file_path = base.train_data
            self.label_file_path = base.train_label
            self.nb_samplings = 5  # int(nb_frames/view_size/batch_size + 1)#*3 for epochFixed *3; for epochFixed2, not.
            self.fb_samplings = 10
        else:
            # /home/yunfei/Data/CVCS/train
            self.file_path = base.val_data
            self.label_file_path = base.val_label
            self.nb_samplings = 1
            self.fb_samplings = 8
        self.batch_size = 1
        # self.cropped_size = [180, 160]
        # self.cropped_size = [300, 200]
        self.cropped_size = [200, 200]

        self.view_size = 5
        self.num_cam = self.view_size
        self.patch_num = 3  # 5
        self.r = 5 # Which represents the real distance of a pixel in the grid of the ground plane.
        # 2 # 0.5m/pixel
        # 10: 0.1m/pixel
        # 5: 0.2m/pixel
        self.grid_reduce = 8  # 0.2/0.025=8 --> 0.025m/pixel
        self.a = 5
        self.b = 5
        # cropped_size, r, a, b, patch_num

        random.seed(2022)

        # list M scenes:
        self.scene_name_list = os.listdir(os.path.dirname(self.file_path))  # 2scenes model training
        self.nb_scenes = len(self.scene_name_list)

    def read_json_frame(self, coords_info, cropped_size, r, a, b, patch_num):

        # set the wld map paras
        wld_map_paras = coords_info['wld_map_paras']
        s, r0, a0, b0, h4, w4, d_delta, d_mean, w_min, h_min = wld_map_paras  # old 640*480 labels

        # reconstruct wld_map_paras:
        w_max = (4 * w4 - 2 * a0) / r0 + w_min
        h_max = (4 * h4 - 2 * b0) / r0 + h_min

        # actual size:
        w_actual = int((w_max - w_min + 2 * a)) * r
        h_actual = int((h_max - h_min + 2 * b)) * r

        h_actual = int(max(h_actual, cropped_size[0] + patch_num))
        w_actual = int(max(w_actual, cropped_size[1] + patch_num))

        # create patch_num patches' bbox coordinates:
        h_range = range(0, h_actual - cropped_size[0] + 1)
        w_range = range(0, w_actual - cropped_size[1] + 1)

        h_random = random.sample(h_range, k=patch_num)
        w_random = random.sample(w_range, k=patch_num)
        hw_random = np.asarray([h_random, w_random])

        wld_map_paras = [r, a, b, h_actual, w_actual, d_mean, w_min, h_min, w_max, h_max]
        coords = coords_info['image_info']
        coords_3d_id_all = []  # np.zeros((1, 3))

        # id = 0
        for point in coords:
            id = point['idx']
            coords_3d_id = point['world'] + [id]
            coords_3d_id_all.append(coords_3d_id)

        coords_3d_id_all = np.asarray(coords_3d_id_all, dtype='float32')
        # form the para list:
        return coords_3d_id_all, wld_map_paras, hw_random

    def read_json_view(self, coords_info):

        # get camera matrix
        cameraMatrix = np.asarray(coords_info['cameraMatrix'])
        fx = cameraMatrix[0][0]
        fy = cameraMatrix[1][1]
        u = cameraMatrix[0][2]
        v = cameraMatrix[1][2]

        # get camera matrix:
        distCoeffs = coords_info['distCoeffs']

        rvec = coords_info['rvec']
        tvec = coords_info['tvec']

        camera_paras = [fx] + [fy] + [u] + [v] + distCoeffs + rvec + tvec
        camera_paras = np.asarray(camera_paras)

        # get people 2D and 3D coords:
        coords = coords_info['image_info']

        coords_2d_all = []  # np.zeros((1, 2))
        coords_3d_id_all = []  # np.zeros((1, 3))

        for point in coords:
            id = point['idx']

            coords_2d0 = point['pixel']
            if coords_2d0 is not None:
                coords_2d = [coords_2d0[1] / 1920.0, coords_2d0[0] / 1080.0]

                coords_3d_id = point['world'] + [id]

                coords_2d_all.append(coords_2d)
                coords_3d_id_all.append(coords_3d_id)

        # form the para list:
        return coords_3d_id_all, coords_2d_all, camera_paras

    def density_map_creation(self, pmap, w, h):
        if pmap.size == 0:
            img_pmap_i = np.zeros((h, w), dtype=np.float32)
            img_dmap = np.asarray(img_pmap_i).astype('f')
        else:
            pmap = np.asarray(pmap)
            img_dmap = np.zeros((h, w))
            x = (pmap[:, 0] * w).astype('int')
            y = (pmap[:, 1] * h).astype('int')
            for k in range(len(x)):
                center_point = (x[k], y[k])
                draw_umich_gaussian(img_dmap, center_point, sigma=self.img_sigma)
        if self.visualize:
            plt.imshow(img_dmap.squeeze())
            plt.show()
        return img_dmap

    def GP_density_map_creation(self, wld_coords, crop_size, wld_map_paras, hw_random):
        # we need to resize the images
        h = int(crop_size[0])
        w = int(crop_size[1])

        r, a, b, h_actual, w_actual, d_mean, w_min, h_min, w_max, h_max = wld_map_paras

        h_actual, w_actual = int(h_actual), int(w_actual)
        patch_num = hw_random.shape[1]

        if wld_coords.size == 0:
            img_pmap_i = np.zeros((h_actual, w_actual))
            GP_density_map = []
            if self.train:
                for p in range(patch_num):
                    hw = hw_random[:, p]
                    GP_density_map_i = img_pmap_i[hw[0]:hw[0] + h, hw[1]:hw[1] + w]
                    GP_density_map.append(GP_density_map_i)
                GP_density_map = torch.from_numpy(np.asarray(GP_density_map)).unsqueeze(1)
            else:
                GP_density_map = torch.from_numpy(img_pmap_i)[None,None]
        else:
            wld_coords_transed = np.zeros(wld_coords.shape, dtype=np.float32)
            wld_coords_transed[:, 0] = (wld_coords[:, 0] - w_min + a) * r
            wld_coords_transed[:, 1] = (wld_coords[:, 1] - h_min + b) * r
            wld_coords_transed = wld_coords_transed.astype('int')

            assert min(wld_coords_transed[:, 0]) >= 0 and max(wld_coords_transed[:, 0]) < w_actual
            assert min(wld_coords_transed[:, 1]) >= 0 and max(wld_coords_transed[:, 1]) < h_actual

            img_pmap = np.zeros((h_actual, w_actual))
            # img_pmap[wld_coords_transed[:, 1], wld_coords_transed[:, 0]] = 1
            for k in range(wld_coords_transed.shape[0]):
                center_point = (wld_coords_transed[k, 0], wld_coords_transed[k, 1])
                draw_umich_gaussian(img_pmap, center_point, sigma=self.map_sigma)

            GP_density_map_0 = img_pmap
            # patch_num = hw_random.shape[1]
            GP_density_map = []
            if self.train:
                for p in range(patch_num):
                    hw = hw_random[:, p]
                    GP_density_map_i = GP_density_map_0[hw[0]:hw[0] + h, hw[1]:hw[1] + w]
                    GP_density_map.append(GP_density_map_i)
                GP_density_map = torch.from_numpy(np.asarray(GP_density_map)).unsqueeze(1).float()  # float32
            else:
                GP_density_map = torch.from_numpy(GP_density_map_0)[None, None]
        return GP_density_map

    def id_unique(self, coords_array):
        # intilize a null list
        coords_array = np.asarray(coords_array)
        unique_list = [[-1, -1, -1, -1]]

        id = coords_array[:, -1]
        n = id.shape[0]
        # traverse for all elements

        for i in range(n):
            id_i = id[i]
            coords_array_i = coords_array[i]

            id_current_unique_list = list(np.asarray(unique_list)[:, -1])
            if id_i not in id_current_unique_list:
                unique_list.append(coords_array_i)

        unique_list = unique_list[1:]
        return unique_list

    def id_diff(self, coords_arrayA, coords_arrayB):  # coords_arrayA is larger
        coords_arrayA = np.asarray(coords_arrayA)
        coords_arrayB = np.asarray(coords_arrayB)

        unique_list = []  # [[-1, -1, -1, -1]]

        if coords_arrayB.size == 0:
            unique_list = coords_arrayA
        else:

            idA = coords_arrayA[:, -1]
            idB = coords_arrayB[:, -1]

            # print(idA)
            # print(idB)

            n = idA.shape[0]
            # traverse for all elements

            for i in range(n):
                id_i = idA[i]
                coords_array_i = coords_arrayA[i]

                # id_current_unique_list = list(np.asarray(unique_list)[:, -1])
                if id_i not in idB:  # id_current_unique_list:
                    unique_list.append(coords_array_i)

            # unique_list = unique_list[1:]
            unique_list = np.asarray(unique_list)
        return unique_list

    def __getitem__(self, index):
        scene_index = int(index / (self.nb_samplings * self.fb_samplings))
        scene_i = self.scene_name_list[scene_index]
        scene_path = os.path.join(self.file_path, scene_i)
        scene_path_label = os.path.join(self.label_file_path, scene_i)

        # list N frames:
        frame_name_list = os.listdir(scene_path_label)  # CSR-net_multi-view_counting_1output_lowRes_5_5_loadWeights
        if self.train:
            frame_j = random.sample(frame_name_list, k=1)[0]
        else:
            frame_index = (index - scene_index * self.nb_samplings * self.fb_samplings) % self.fb_samplings
            frame_j = frame_name_list[frame_index]

        # select views first:
        frame_0 = '0'
        label_path0 = os.path.join(self.label_file_path, scene_i, frame_0, 'json_paras/')
        label_path_list0 = os.listdir(label_path0)
        label_path_list_sampling = random.sample(label_path_list0, k=self.view_size)
        # label_path_list_sampling = ['27.json']

        # 每个场景下的视角数不同，但都大于94
        if scene_i == "scene_24" and int(frame_j) > 90:
            frame_j = str(random.randint(0, 90))

        frame_path = os.path.join(scene_path, frame_j)

        img_path = frame_path + '/jpgs/'
        label_path = os.path.join(self.label_file_path, scene_i, frame_j, 'json_paras/')

        # decide the whole crowd GP density maps of the frame:
        # read all people
        label_name0 = label_path_list0[0]
        label_path_name0 = os.path.join(label_path, label_name0)

        # get the world parameters
        with open(label_path_name0, 'r') as data_file:
            coords_info_frame = json.load(data_file)
        coords_3d_id_all_frame, wld_map_paras_frame, hw_random = self.read_json_frame(coords_info_frame,
                                                                                      self.cropped_size,
                                                                                      self.r, self.a, self.b,
                                                                                      self.patch_num)
        wld_map_paras_frame = np.asarray(wld_map_paras_frame)

        hw_random = np.asarray(hw_random)
        wld_map_paras = wld_map_paras_frame

        img_views = []
        camera_paras = []
        wld_coords = []

        single_view_dmaps = []
        GP_view_dmaps = []

        # get the images and ground truths of all cameras
        for label_name in label_path_list_sampling:
            img_name = label_name[0:-5] + '.jpg'

            # read images
            img = Image.open(os.path.join(img_path, img_name))
            if self.transform:
                img = self.transform(img)
            img_views.append(img)

            # read labels
            label_path_name = os.path.join(label_path, label_name)
            with open(label_path_name, 'r') as data_file:
                coords_info = json.load(data_file)
            coords, coords_2d, paras = self.read_json_view(coords_info)
            # generate the ground plane density map of single-view.
            coords_GP = np.asarray(coords)
            GP_view_dmaps_i = self.GP_density_map_creation(coords_GP,
                                                           self.cropped_size,
                                                           wld_map_paras,
                                                           hw_random) * self.facofmaxgt
            GP_view_dmaps.append(GP_view_dmaps_i)

            coords_2d = np.asarray(coords_2d)
            single_view_dmaps_i = self.density_map_creation(coords_2d, w=self.upsample_shape[1],
                                                            h=self.upsample_shape[0])
            single_view_dmaps_i = np.expand_dims(single_view_dmaps_i, axis=0)
            single_view_dmaps.append(torch.from_numpy(single_view_dmaps_i))

            # form the camera paras list
            camera_paras.append(paras)

            # get the wld_coords:
            wld_coords = wld_coords + coords
        GP_view_dmaps = torch.stack(GP_view_dmaps, dim=1)  # [patch, view, channel=1, crop_h, crop_w]
        # All the world coordinates except the duplicated ID
        wld_coords = self.id_unique(wld_coords)
        wld_coords = np.asarray(wld_coords)

        # create the gt dmap of ground plane
        GP_density_map = self.GP_density_map_creation(wld_coords,
                                                      self.cropped_size,
                                                      wld_map_paras,
                                                      hw_random) * self.facofmaxgt

        camera_paras = torch.from_numpy(np.asarray(camera_paras))
        img_views = torch.stack(img_views)
        imgs_gt = torch.stack(single_view_dmaps, dim=0).float() * self.facofmaxgt

        return img_views, imgs_gt, camera_paras, torch.from_numpy(wld_map_paras), torch.from_numpy(
            hw_random), GP_density_map, GP_view_dmaps.float()

    def __len__(self):
        if self.train:
            len_num = self.nb_scenes * self.fb_samplings * self.nb_samplings  # 23*5*10
        else:
            len_num = self.nb_scenes * self.fb_samplings * self.nb_samplings  # 8*1*10
        return len_num


def test():
    from datasets.CVCS.CVCS import CVCS
    from torch.utils.data import DataLoader
    root = '/mnt/data/Datasets/CVCS'
    # 使用ImageNet的均值和标准差进行归一化
    _transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    base = CVCS(root)
    train_set = frameDataset(base, train=True, _transform=_transform)
    # img_views, single_view_dmaps, camera_paras, wld_map_paras, hw_random, GP_density_map = train_set.__getitem__(0)
    val_set = frameDataset(base, train=False, _transform=_transform)
    # loop check
    # for i in range(train_set.__len__()):
    #     train_set.__getitem__(i)
    # train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=2)
    # img_views, single_view_dmaps, camera_paras, wld_map_paras, hw_random, GP_density_map, GP_view_dmaps = next(
    #     iter(train_loader))
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)
    for i, data in enumerate(val_loader):
        pass
    img_views, single_view_dmaps, camera_paras, wld_map_paras, hw_random, GP_density_map, GP_view_dmaps = next(
        iter(val_loader))
    # print()
    print(img_views.shape)
    # print('GP_density_map shape', GP_density_map.shape)
    # for i in range(5):
    #     plt.imshow()


if __name__ == '__main__':
    test()
