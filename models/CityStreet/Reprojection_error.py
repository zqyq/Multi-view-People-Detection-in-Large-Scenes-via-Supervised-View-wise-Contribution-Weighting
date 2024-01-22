import json
import os
from math import sqrt, pow

import h5py
import torch

from models.CityStreet import SpatialTransformer_v3
from utils.person_help import *


def dist_error(proj_coords, gp_coord):
    view_error = 0
    id_count = 0
    for view in range(3):
        if proj_coords[view] is not None:
            id_count += 1
            view_error += sqrt(pow(proj_coords[view][0] - gp_coord[0], 2) + pow(proj_coords[view][1] - gp_coord[1], 2))
    return view_error / (id_count + 1e-12)


def find_coord(input_tensor: torch.Tensor):
    proj_coords = {}
    input_tensor = input_tensor.detach().cpu()
    for view in range(3):
        if input_tensor[view].max() > 0:
            res = (input_tensor == input_tensor[view].max()).nonzero()
            proj_coords[view] = tuple(map(lambda x: float(x.item()), res[0][-2:]))
        else:
            proj_coords[view] = None
    return proj_coords

def citystreet(device):


    STN = SpatialTransformer_v3(input_size=[1, 380, 676, 1], output_size=(768, 640), device=device,
                                person_heights=[1750])
    # ts_ones = torch.zeros(3, 1, 380, 676)
    #     # ts_ones[0][0][100][200] = 10
    #     # ts_ones[2][0][100][200] = 10
    #     # ts_ones[1][0][100][200] = 10
    #     # proj_ts = STN(ts_ones.permute(0, 2, 3, 1), height=1750).permute(0, 3, 1, 2)
    #     # pc = find_coord(proj_ts)

    json_files_dir = '/home/yunfei/Data/CityStreet/labels'
    data = {}

    # with open(os.path.join(json_files_dir, 'GP_pmap_height.json')) as file:
    #     data['GP'] = json.load(file)
    with open(os.path.join(json_files_dir, 'via_region_data_view1.json')) as file:
        data['view1'] = json.load(file)
    with open(os.path.join(json_files_dir, 'via_region_data_view2.json')) as file:
        data['view2'] = json.load(file)
    with open(os.path.join(json_files_dir, 'via_region_data_view3.json')) as file:
        data['view3'] = json.load(file)

    frame_range = range(636, 1406, 2)
    error = 0
    with h5py.File('/home/yunfei/Data/CityStreet/Street_groundplane_pmap.h5', 'r') as f:
        for frame in frame_range:
            frame_error = 0
            frame_part = f['v_pmap_GP'][f['v_pmap_GP'][:, 0] == frame, :]
            for i in range(frame_part.shape[0]):
                id = str(int(frame_part[i, 1]))
                real_x = frame_part[i, 2]
                real_y = frame_part[i, 3]
                # gp_region = data['GP']['frame_{:0>4}.jpg'.format(frame)]['regions']
                # 地面坐标
                # real_x = gp_region[id]['shape_attributes']['cx']
                # real_y = gp_region[id]['shape_attributes']['cy']
                # if real_x is None or real_y is None:
                #     continue
                # real_height = gp_region[id]['shape_attributes']['height']
                real_coord = (real_y, real_x)
                stn_input = torch.zeros(3, 1, 380, 676).to(device)
                # id_occur = 0  # 该ID人出现在3个视角里次数
                for view in range(3):
                    view_region = data[f'view{view + 1}']['frame_{:0>4}.jpg'.format(frame)]['regions']
                    if id in view_region.keys():  # 地面人位于该视角内.
                        # 图像坐标
                        img_x = view_region[id]['shape_attributes']['cx']
                        img_y = view_region[id]['shape_attributes']['cy']
                        if img_x is not None and img_y is not None:
                            img_x = int(img_x / 4)
                            img_y = int(img_y / 4)
                            if 0 <= img_x < 676 and 0 <= img_y < 380:
                                stn_input[view][0][img_y][img_x] = 10
                # 投影坐标
                input_tensor = STN(stn_input.permute(0, 2, 3, 1), height=1750).permute(0, 3, 1, 2)
                proj_coords = find_coord(input_tensor)
                view_error = dist_error(proj_coords, real_coord)
                frame_error += view_error
            frame_error /= frame_part.shape[0]
            error += frame_error
            index = int((frame - 634) / 2)
            print(f'frame={frame},error:{error / index:.3f}')
        error /= len(frame_range)
        print(f'Average reprojection error is:{error}')
        
def wildtrack(device):
    img_shape, reducedgrid_shape = dataset.img_shape, dataset.reducedgrid_shape
    imgcoord2worldgrid_matrices = get_imgcoord2worldgrid_matrices(dataset.base.intrinsic_matrices,
                                                                       dataset.base.extrinsic_matrices,
                                                                       dataset.base.worldgrid2worldcoord_mat)
    coord_map = create_coord_map(reducedgrid_shape + [1])

    # img
    upsample_shape = list(map(lambda x: int(x / dataset.img_reduce), img_shape))
    img_reduce = np.array(img_shape) / np.array(upsample_shape)
    img_zoom_mat = np.diag(np.append(img_reduce, [1]))
    # map
    map_zoom_mat = np.diag(np.append(np.ones([2]) / dataset.grid_reduce, [1]))
    # projection matrices: img feat -> map feat
    proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
                      for cam in range(num_cam)]
if __name__ == '__main__':
    device = 'cuda:0'
    citystreet(device)
    wildtrack(device)
