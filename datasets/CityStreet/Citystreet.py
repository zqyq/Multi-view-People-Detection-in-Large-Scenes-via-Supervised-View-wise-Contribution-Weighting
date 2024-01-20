import os
import re

import PIL.Image
import numpy as np
import torch
from torchvision.datasets import VisionDataset


class Citystreet(VisionDataset):
    def __init__(self, root):
        super().__init__(root)
        self.root = root
        self.__name__ = 'Citystreet'
        self.img_shape, self.worldgrid_shape = [1520, 2704], [768, 640]  # H,W; N_row,N_col
        # self.cam_range = [1, 3, 4]
        self.num_frame = 1500
        self.num_cam = 3

        # x,y actually means i,j in CityStreet, which correspond to h,w
        # self.indexing = 'xy'
        # #  for world map indexing

    # 注意frame_range belongs to Training: frame_0636.jpg---frame_1234.jpg, 300 images in total.
    # Testing: frame_1236.jpg---frame_1634.jpg, 200 images in total.
    # def
    def get_img_fpath(self):
        viewimg_fpaths = {view: {} for view in range(1, 4)}
        for view, camera_folder in enumerate(sorted(os.listdir(os.path.join(self.root, 'image_frames')))):
            for fname in sorted(os.listdir(os.path.join(self.root, 'image_frames', camera_folder))):
                frame = int(fname.split('_')[1].split('.')[0])
                if frame in range(636, 1636, 2):
                    viewimg_fpaths[view + 1][frame] = os.path.join(self.root, 'image_frames', camera_folder, fname)
        return viewimg_fpaths


def test():
    from torchvision.transforms import ToTensor
    dataset = Citystreet(os.path.expanduser('~/Data/CityStreet'))
    # frame_range = range(636, 1236, 2)
    imgs_fpaths = dataset.get_img_fpath()
    # print(imgs_fpaths[1][636])
    # x = PIL.Image.open(imgs_fpaths[1][636]).convert('RGB')
    transform = ToTensor()
    # x = transform(x)
    data_sum, data_squared_sum = 0, 0
    for view in range(1, 4):
        for frame in range(636, 1236, 2):
            img = PIL.Image.open(imgs_fpaths[view][frame]).convert('RGB')
            img = transform(img)
            data_sum += torch.mean(img, dim=[1, 2])
            data_squared_sum += torch.mean(img ** 2, dim=[1, 2])
    city_mean = data_sum / 900
    city_std = (data_squared_sum / 900 - city_mean ** 2) ** 0.5
    print(city_mean, city_std)


if __name__ == '__main__':
    test()
