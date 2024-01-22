import datetime
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets.CityStreet import camera_proj_Zhang as proj

device = torch.device('cpu')


def world2image(view, worldcoords):
    N = worldcoords.shape[0]
    imgcoords = []
    for i in range(N):
        worldcoord = worldcoords[i, :]

        Xw = worldcoord[0].item()
        Yw = worldcoord[1].item()
        Zw = worldcoord[2].item()

        XYi = proj.World2Image(view, Xw, Yw, Zw)
        imgcoords.append(XYi)
    imgcoords = np.asarray(imgcoords)
    return imgcoords


# def getall_iniform(input_shape, out_shape, depth_scales, devicenum):
#     view_gp_masks = []
#     # input_shape = (1, 1520, 2704, 3)，因为在网络中需要对img_feature进行投影，所以这里的输入维度应该为[1, 380, 676, 1]
#     # input_shape = [1, 380, 676, 1]
#     proj_views_heights, view_gp_masks = proj_2Dto2D(input_shape, out_shape, depth_scales, devicenum)
#     return proj_views_heights, view_gp_masks


def proj_2Dto2D(input_shape, out_shape, devicenum, person_heights, proj_store_path):
    w = 676
    h = 380
    H = out_shape[0]
    W = out_shape[1]
    bbox = [352 * 0.8, 522 * 0.8]
    resolution_scaler = 76.25
    proj_views_heights = {}
    view_gp_masks = []
    # bbox = [bbox[0] * fac, bbox[1] * fac]
    # bbox = []
    # ph = 1.75 * 1000  # average height of a person in millimeters
    # person_heights=list(map(lambda x: int(x), np.linspace(1600, 1900, self.depth_scales)))
    current_heights = person_heights
    # nR = input_shape[0]
    fh = input_shape[1]
    fw = input_shape[2]
    # fdim = input_shape[3]
    # batch_size, gp_x, gp_y = nR, W, H
    rsz_h = float(fh) / (h * 4)
    rsz_w = float(fw) / (w * 4)
    # Create voxel grid
    fac = float(out_shape[0] / 192)
    grid_fac = 4 / fac

    # create meshgrid
    grid_rangeX = np.linspace(0, W - 1, W)
    grid_rangeY = np.linspace(0, H - 1, H)
    # grid_rangeZ = hi # np.linspace(0, D - 1, D)
    # grid_rangeX, grid_rangeY, grid_rangeZ = np.meshgrid(grid_rangeX, grid_rangeY, grid_rangeZ)
    grid_rangeX, grid_rangeY = np.meshgrid(grid_rangeX, grid_rangeY)
    # #print("grid_rangeX",grid_rangeX)
    grid_rangeX = np.reshape(grid_rangeX, [-1])
    grid_rangeY = np.reshape(grid_rangeY, [-1])
    # grid_rangeZ = np.reshape(grid_rangeZ, [-1])

    grid_rangeX = (grid_rangeX * grid_fac - bbox[0]) * resolution_scaler
    grid_rangeX = grid_rangeX + 1
    grid_rangeX = np.expand_dims(grid_rangeX, 1)

    grid_rangeY = (grid_rangeY * grid_fac - bbox[1]) * resolution_scaler
    grid_rangeY = grid_rangeY + 1
    grid_rangeY = np.expand_dims(grid_rangeY, 1)
    for view in range(1, 4):
        # view_gp_masks
        viewnum = view
        if view == 1:
            view = 'view1'
        #     view_gp_mask = np.load('/home/yunfei/Data/CityStreet/mask/view1_GP_mask.npz')
        if view == 2:
            view = 'view2'
        #     view_gp_mask = np.load('/home/yunfei/Data/CityStreet/mask/view2_GP_mask.npz')
        if view == 3:
            view = 'view3'
        #     view_gp_mask = np.load('/home/yunfei/Data/CityStreet/mask/view3_GP_mask.npz')
        # # gp view mask:
        # view_gp_mask = view_gp_mask['arr_0']
        # view_gp_mask = cv2.resize(view_gp_mask, (out_shape[1], out_shape[0]))
        # view_gp_mask = torch.from_numpy(view_gp_mask).float()
        # view_gp_mask = torch.unsqueeze(view_gp_mask, 0)
        # view_gp_mask = torch.unsqueeze(view_gp_mask, -1)
        # # num_channels = fdim  ###### remember to add the depth dim
        # # view_gp_mask = view_gp_mask.repeat(1, 1, 1, num_channels)
        # view_gp_masks.append(view_gp_mask)
        # np.save(os.path.join(
        #         '/mnt/data/Yunfei/Study/Baseline_MVDet/Projection_files/Single_view_height_projeciton_relation',
        #         f'proj_{view}_mask.npy'), view_gp_mask.numpy())        # proj_views_heights

        for ph in current_heights:
            grid_rangeZ = ph * np.ones(grid_rangeX.shape)
            # #print("grid_rangeZ",grid_rangeZ.size())
            wldcoords = np.concatenate(([grid_rangeX, grid_rangeY, grid_rangeZ]), axis=1)
            view_ic = world2image(view, wldcoords)
            view_ic = np.transpose(view_ic)
            view_ic[0:1, :] = view_ic[0:1, :] * rsz_w
            view_ic[1:2, :] = view_ic[1:2, :] * rsz_h
            view_ic[2:3, :] = view_ic[2:3, :]  # / 400
            proj_height = np.concatenate([view_ic[0:1, :], view_ic[1:2, :], view_ic[2:3, :]], axis=0)
            proj_views_heights[(viewnum, int(ph))] = torch.from_numpy(proj_height)
            np.save(os.path.join(proj_store_path,
                                 f'proj_{view}_height{int(ph)}_reso{str(out_shape)}.npy'), view_ic)
    return proj_views_heights


def generate_cor_mask(out_shape, data_root, mask_store_path):
    # view_gp_masks=[]
    for view in range(1, 4):
        # view_gp_masks
        if view == 1:
            view = 'view1'
            view_gp_mask = np.load(os.path.join(data_root, 'CityStreet/mask/view1_GP_mask.npz'))
        if view == 2:
            view = 'view2'
            view_gp_mask = np.load(os.path.join(data_root, 'CityStreet/mask/view2_GP_mask.npz'))
        if view == 3:
            view = 'view3'
            view_gp_mask = np.load(os.path.join(data_root, 'CityStreet/mask/view3_GP_mask.npz'))
        # gp view mask:
        view_gp_mask = view_gp_mask['arr_0']
        view_gp_mask = cv2.resize(view_gp_mask, (out_shape[1], out_shape[0]))
        view_gp_mask = torch.from_numpy(view_gp_mask).float()
        view_gp_mask = torch.unsqueeze(view_gp_mask, 0)
        view_gp_mask = torch.unsqueeze(view_gp_mask, -1)
        # num_channels = fdim  ###### remember to add the depth dim
        # view_gp_mask = view_gp_mask.repeat(1, 1, 1, num_channels)
        # view_gp_masks.append(view_gp_mask)
        np.save(os.path.join(mask_store_path,
                             f'proj_{view}_mask_outsize{out_shape}.npy'), view_gp_mask.numpy())


def test():
    input_imgshape = (1, 1520, 2704, 3)
    outshape = (768, 640)
    depth_scales = 4
    print(datetime.datetime.now())
    t0 = time.time()
    devicenum = 'cuda:0'
    proj_views_heights, view_gp_masks = proj_2Dto2D(input_imgshape, outshape, depth_scales, devicenum)
    t1 = time.time()
    print('t1 - t0', t1 - t0)
    print(datetime.datetime.now())
    print(proj_views_heights.keys())
    for key in proj_views_heights.keys():
        print(key, proj_views_heights[key].shape)

    print(view_gp_masks[0].shape)


if __name__ == '__main__':
    test()
