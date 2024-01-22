# from keras.layers.core import Layer
import torch

import torch.nn as nn

# from processing_layer import Cropping_layer
import numpy as np


def spatial_transoformation_layer(paras, x):
    output = _transform(paras, x)
    output = output[0].clone().permute((0, 3, 1, 2))
    return output


def _transform(paras, inputs):
    # batch_size0 = paras[0]
    view_size = paras[1]
    patch_num = paras[2]
    cropped_size = paras[3]

    input_feature, input_camera_paras, input_wld_map_paras, input_hw_random = inputs
    input_feature = input_feature.clone().permute(dims=(0, 2, 3, 1))  # [B,H,W,C]

    # batch_size = tf.shape(input_feature)[0]
    # height = tf.shape(input_feature)[1]
    # width = tf.shape(input_feature)[2]
    # num_channels = tf.shape(input_feature)[3]

    batch_size = input_feature.shape[0]  # actually, equals b*v
    height = input_feature.shape[1]
    width = input_feature.shape[2]
    num_channels = input_feature.shape[3]


    output_size = cropped_size
    output_height = output_size[0]
    output_width = output_size[1]

    transformed_image_all = torch.zeros([1, patch_num, output_height, output_width, num_channels]).to(
        input_feature.device)
    for i in range(batch_size):

        # set indices_grid:
        batch_id = int(i / view_size)  # actually, equals b*v
        input_wld_map_paras_i = input_wld_map_paras[batch_id, :]

        transformed_image_p = torch.zeros([1, output_height, output_width, num_channels]).to(input_feature.device)
        for p in range(patch_num):
            hw = input_hw_random[batch_id, :, p]

            indices_grid = _meshgrid(output_height, output_width, hw).to(input_feature.device)
            indices_grid = torch.unsqueeze(indices_grid, dim=0)
            indices_grid = torch.reshape(indices_grid, [-1])  # flatten?

            # indices_grid = indices_grid.repeat(torch.stack([1]))
            indices_grid = torch.reshape(indices_grid, (1, 3, -1))

            transformed_grid = _create_transformed_grid(indices_grid.to(input_feature.device),
                                                        input_camera_paras[i].to(input_feature.device),
                                                        input_wld_map_paras_i.to(input_feature.device),
                                                        [height, width]).to(input_feature.device)

            x_s = transformed_grid[:, 0, :]  # tf.slice(transformed_grid, [0, 0, 0], [-1, 1, -1])
            y_s = transformed_grid[:, 1, :]  # tf.slice(transformed_grid, [0, 1, 0], [-1, 1, -1])
            x_s_flatten = torch.reshape(x_s, [-1])
            y_s_flatten = torch.reshape(y_s, [-1])

            transformed_image = _interpolate(input_feature[i:i + 1],
                                             x_s_flatten,
                                             y_s_flatten,
                                             output_size).to(input_feature.device)

            transformed_image = torch.reshape(transformed_image, shape=(1,
                                                                        output_height,
                                                                        output_width,
                                                                        num_channels))
            transformed_image_p = torch.cat([transformed_image_p, transformed_image], dim=0)
        transformed_image_p = transformed_image_p[1:]
        transformed_image_p = torch.unsqueeze(transformed_image_p, dim=0)
        transformed_image_all = torch.cat([transformed_image_all, transformed_image_p], dim=0)
    transformed_image_all = transformed_image_all[1:].to(input_feature.device)

    return transformed_image_all


def _repeat(x, num_repeats):
    ones = torch.ones((1, num_repeats))
    x = torch.reshape(x, shape=(-1, 1))
    x = torch.matmul(x, ones)
    return torch.reshape(x, [-1])


def _interpolate(image, x, y, output_size):
    # batch_size = tf.shape(image)[0]
    # height = tf.shape(image)[1]
    # width = tf.shape(image)[2]
    # num_channels = tf.shape(image)[3]

    batch_size = image.shape[0]
    height = image.shape[1]
    width = image.shape[2]
    num_channels = image.shape[3]

    # x = torch.tensor(x)
    # y = torch.tensor(y)

    # height_float = torch.cast(height, dtype='float32')
    # width_float = torch.cast(width, dtype='float32')

    height_float = torch.tensor(height)
    width_float = torch.tensor(width)

    output_height = output_size[0]
    output_width = output_size[1]

    x = .5 * (x + 1.0) * width_float
    y = .5 * (y + 1.0) * height_float

    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1

    max_y = height - 1
    max_x = width - 1
    # zero = torch.zeros([1])

    x0 = torch.clamp(x0, 0, max_x).to(image.device)
    x1 = torch.clamp(x1, 0, max_x).to(image.device)
    y0 = torch.clamp(y0, 0, max_y).to(image.device)
    y1 = torch.clamp(y1, 0, max_y).to(image.device)

    flat_image_dimensions = width * height
    pixels_batch = torch.range(0, batch_size - 1) * flat_image_dimensions
    flat_output_dimensions = output_height * output_width
    base = _repeat(pixels_batch, flat_output_dimensions).to(image.device)
    base_y0 = base + y0 * width
    base_y1 = base + y1 * width
    indices_a = base_y0 + x0
    indices_b = base_y1 + x0
    indices_c = base_y0 + x1
    indices_d = base_y1 + x1

    flat_image = torch.reshape(image, shape=(-1, num_channels))
    # flat_image = torch.tensor(flat_image)
    pixel_values_a = torch.index_select(flat_image, dim=0, index=indices_a.long())
    pixel_values_b = torch.index_select(flat_image, dim=0, index=indices_b.long())
    pixel_values_c = torch.index_select(flat_image, dim=0, index=indices_c.long())
    pixel_values_d = torch.index_select(flat_image, dim=0, index=indices_d.long())

    # x0 = torch.cast(x0, 'float32')
    # x1 = torch.cast(x1, 'float32')
    # y0 = torch.cast(y0, 'float32')
    # y1 = torch.cast(y1, 'float32')

    area_a = torch.unsqueeze(((x1 - x) * (y1 - y)), 1)
    area_b = torch.unsqueeze(((x1 - x) * (y - y0)), 1)
    area_c = torch.unsqueeze(((x - x0) * (y1 - y)), 1)
    area_d = torch.unsqueeze(((x - x0) * (y - y0)), 1)

    output = area_a * pixel_values_a + area_b * pixel_values_b + area_c * pixel_values_c + area_d * pixel_values_d
    return output


def _meshgrid(height, width, offset):
    # height = int(height)
    # width = int(width)

    # x_linspace = tf.linspace(-1., 1., width)
    # y_linspace = tf.linspace(-1., 1., height)

    x_linspace = torch.linspace(offset[1], offset[1] + width - 1, width)
    y_linspace = torch.linspace(offset[0], offset[0] + height - 1, height)

    # x_coordinates, y_coordinates = torch.meshgrid(x_linspace, y_linspace)
    y_coordinates, x_coordinates = torch.meshgrid(y_linspace, x_linspace)

    x_coordinates = torch.reshape(x_coordinates, [-1])
    y_coordinates = torch.reshape(y_coordinates, [-1])
    ones = torch.ones_like(x_coordinates)
    indices_grid = torch.cat([x_coordinates, y_coordinates, ones], 0)
    return indices_grid


def _projectPoints(
        wld_coords_transed,
        rvec,
        tvec,
        fx, fy, u, v,
        distCoeffs):
    # wld
    Xw = wld_coords_transed[:, 0:1, :]
    Yw = wld_coords_transed[:, 1:2, :]
    Zw = wld_coords_transed[:, 2:, :]

    # get the rotaion matrix:
    theta = torch.norm(rvec)
    rvec = torch.div(rvec, theta)
    rx, ry, rz = rvec[0], rvec[1], rvec[2]
    skew = torch.tensor([[0, -rz, ry],
                         [rz, 0, -rx],
                         [-ry, rx, 0]]).to(wld_coords_transed.device)  # , dtype='float32')
    rvec = torch.unsqueeze(rvec, dim=1)
    r1 = torch.cos(theta) * torch.eye(3).to(wld_coords_transed.device)
    r2 = (torch.tensor([1]).to(wld_coords_transed.device) - torch.cos(theta)) * torch.matmul(rvec,
                                                                                             torch.transpose(rvec, 0,
                                                                                                             1))
    r3 = torch.sin(theta) * skew
    R = r1 + r2 + r3
    mR11, mR12, mR13, mR21, mR22, mR23, mR31, mR32, mR33 = R[0][0], R[0][1], R[0][2], \
                                                           R[1][0], R[1][1], R[1][2], \
                                                           R[2][0], R[2][1], R[2][2]

    mTx, mTy, mTz = tvec[0], tvec[1], tvec[2]

    # step 1:
    # RX + T
    xc = mR11 * Xw + mR12 * Yw + mR13 * Zw + mTx
    yc = mR21 * Xw + mR22 * Yw + mR23 * Zw + mTy
    zc = mR31 * Xw + mR32 * Yw + mR33 * Zw + mTz

    # step 2:
    Xu0 = torch.div(xc, zc + 1.0 / 19200.0).to(wld_coords_transed.device)  # avoid zero
    Yu0 = torch.div(yc, zc + 1.0 / 10800.0).to(wld_coords_transed.device)

    # step 3:
    # distCoeffs = [k1, k2, p1, p2, k3]
    k1, k2, p1, p2, k3 = distCoeffs[0], distCoeffs[1], distCoeffs[2], \
                         distCoeffs[3], distCoeffs[4]
    r = torch.sqrt(Xu0 * Xu0 + Yu0 * Yu0)
    beta = (1 + k1 * torch.pow(r, 2) + k2 * torch.pow(r, 4) + k3 * torch.pow(r, 6)) / 1

    Xu1 = Xu0 * beta + 2 * p1 * Xu0 * Yu0 + p2 * (torch.pow(r, 2) + 2 * Xu0 * Xu0)
    Yu1 = Yu0 * beta + 2 * p2 * Xu0 * Yu0 + p1 * (torch.pow(r, 2) + 2 * Yu0 * Yu0)

    Xu2 = (fx * Xu1 + u) * 1920
    Yu2 = (fy * Yu1 + v) * 1080

    img_coords = torch.cat((Xu2, Yu2), dim=1)

    return img_coords


def _create_transformed_grid(
        indices_grid,
        input_camera_paras,
        input_wld_map_paras,
        image_size):
    # define camera paras
    height, width = image_size
    fx, fy, u, v = input_camera_paras[0], input_camera_paras[1], \
                   input_camera_paras[2], input_camera_paras[3]

    # cameraMatrix = tf.cast([[fx, 0, u],
    #                         [0, fy, v],
    #                         [0, 0, 1]], dtype='float32')

    distCoeffs = input_camera_paras[4:4 + 5]
    rvec = input_camera_paras[4 + 5:4 + 5 + 3]
    tvec = input_camera_paras[4 + 5 + 3:]

    # define wld map paras
    # s, r, w_max, h_max, h, w, d_delta, d_mean, w_min, h_min = input_wld_map_paras[0], input_wld_map_paras[1], input_wld_map_paras[2], \
    #                                                   input_wld_map_paras[3], input_wld_map_paras[4], input_wld_map_paras[5], \
    #                                                   input_wld_map_paras[6], input_wld_map_paras[7], input_wld_map_paras[8], \
    #                                                   input_wld_map_paras[9]
    r, a, b, h_actual, w_actual, d_mean, w_min, h_min, w_max, h_max = input_wld_map_paras[0].to(indices_grid.device), \
                                                                      input_wld_map_paras[1].to(indices_grid.device), \
                                                                      input_wld_map_paras[2].to(indices_grid.device), \
                                                                      input_wld_map_paras[3].to(indices_grid.device), \
                                                                      input_wld_map_paras[4].to(indices_grid.device), \
                                                                      input_wld_map_paras[5].to(indices_grid.device), \
                                                                      input_wld_map_paras[6].to(indices_grid.device), \
                                                                      input_wld_map_paras[7].to(indices_grid.device), \
                                                                      input_wld_map_paras[8].to(indices_grid.device), \
                                                                      input_wld_map_paras[9].to(indices_grid.device)

    # r = self.resize
    # a = int(r/2)
    # b = int(r/2)
    #
    # # actual size:
    # w_actual = tf.cast((w_max-w_min)*r + 2*a, tf.int32)
    # h_actual = tf.cast((h_max-h_min)*r + 2*b, tf.int32)

    wld_coords_transed_w = (indices_grid[:, 0:1, :] * 1) / r + w_min - a  # output size deicides this
    wld_coords_transed_h = (indices_grid[:, 1:2, :] * 1) / r + h_min - b
    wld_coords_transed_d = indices_grid[:, 2:, :] * d_mean
    wld_coords_transed = torch.cat([wld_coords_transed_w, wld_coords_transed_h, wld_coords_transed_d],
                                   dim=1)

    transformed_grid = _projectPoints(wld_coords_transed,
                                      rvec,
                                      tvec,
                                      fx, fy, u, v,
                                      distCoeffs)
    # image coords, resize to 1/3 of the original resolution.
    # transformed_grid = transformed_grid/3.0

    # transformed_grid = tf.transpose(transformed_grid, [0, 2, 1]) # input features dicide width and height
    transformed_grid_w = (transformed_grid[:, 0:1, :] * 2.0) / 1920 - 1.0  # original resolution 1920*1080
    transformed_grid_h = (transformed_grid[:, 1:, :] * 2.0) / 1080 - 1.0

    transformed_grid_w = torch.clamp(transformed_grid_w, -10, 10)
    transformed_grid_h = torch.clamp(transformed_grid_h, -10, 10)

    transformed_grid_wh = torch.cat([transformed_grid_w, transformed_grid_h], dim=1)
    return transformed_grid_wh
