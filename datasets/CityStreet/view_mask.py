import os.path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def get_view_gp_mask(root, view, outshape=None):
    view_gp_mask = np.load(os.path.join(root, f'mask/view{view}_GP_mask.npz'))
    # gp view mask:
    view_gp_mask = view_gp_mask['arr_0']
    view_gp_mask = cv2.resize(view_gp_mask, (outshape[1], outshape[0]))
    # view_gp_mask = torch.from_numpy(view_gp_mask).float()
    return view_gp_mask


def getCameraMask():
    cam_masks = []
    cam_mask_dir = ['/home/yunfei/Data/CityStreet/ROI_maps/ROIs/camera_view/mask1_ic.npz',
                    '/home/yunfei/Data/CityStreet/ROI_maps/ROIs/camera_view/mask2_ic.npz',
                    '/home/yunfei/Data/CityStreet/ROI_maps/ROIs/camera_view/mask2_ic.npz']
    for view in range(len(cam_mask_dir)):
        data = np.load(cam_mask_dir[view])
        mask = data[view]['arr_0']
        cam_masks.append(mask)
    cam_masks = np.stack(cam_masks)


def under_cropped_mask(num_cam, outshape, hw_random, cropped_size, visualize=False):
    patch_view_mask = {}
    # hw_random [1, p_num, 2]
    for p in range(hw_random.shape[1]):
        hw = hw_random[p]
        start_coord_h = hw[0]
        start_coord_w = hw[1]
        c_h = cropped_size[0]
        c_w = cropped_size[1]
        for view in range(1, num_cam + 1):
            view_mask = get_view_gp_mask(view, outshape)
            patch_view_mask[(p, view)] = view_mask[start_coord_h:start_coord_h + c_h,
                                         start_coord_w:start_coord_w + c_w]
            # visualize
            if visualize:
                plt.imshow(patch_view_mask[(p, view)])
                plt.colorbar()
                plt.show()
    return patch_view_mask


if __name__ == '__main__':
    outshape = (768, 640)
    # vp = get_view_gp_mask(1, outshape)
    hw_random = [[100, 200], [100, 250]]
    cropped_size = [200, 200]
    p_v_m = under_cropped_mask(3, outshape, hw_random, cropped_size)

    print(1)
