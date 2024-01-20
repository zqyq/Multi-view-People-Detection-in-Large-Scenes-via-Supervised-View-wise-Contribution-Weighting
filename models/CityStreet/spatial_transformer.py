# Pytorch
import datetime
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from PIL import Image
from multiview_detector.utils.person_help import vis
from multiview_detector.models.CityStreet import Ini_reduce_time_mul_height as Inir


class SpatialTransformer_v3:
    def __init__(self,
                 input_size,
                 output_size,
                 device,
                 **kwargs):
        # self.view = view
        self.output_size = output_size
        self.input_size = input_size
        self.devicenum = device
        # self.person_heights = list(map(lambda x: int(x), np.linspace(1600, 1900, self.depth_scales)))
        self.person_heights = kwargs['person_heights']
        self.proj_views_heights = {}
        self.view_gp_masks = []
        # for direct use of projection instaed of calculating every time.
        for ph in self.person_heights:
            if not os.path.exists(os.path.join(
                    '/mnt/data/Yunfei/Study/Baseline_MVDet/Projection_files/Single_view_height_projeciton_relation',
                    f'proj_view1_height{int(ph)}_reso{str(output_size)}.npy')):
                Inir.proj_2Dto2D(self.input_size, self.output_size, self.devicenum, [ph])
            for view in range(1, 4):
                self.proj_views_heights[(view, int(ph))] = torch.from_numpy(np.load(
                    os.path.join(
                        '/mnt/data/Yunfei/Study/Baseline_MVDet/Projection_files/Single_view_height_projeciton_relation',
                        f'proj_view{view}_height{int(ph)}_reso{str(output_size)}.npy'))).to(
                    self.devicenum)

        if not os.path.exists(f'/mnt/data/Yunfei/Study/Baseline_MVDet/Projection_files/Single_view_height'
                              f'_projeciton_relation/proj_viewview1_mask_outsize{output_size}.npy'):
            Inir.generate_cor_mask(out_shape=self.output_size)

        for view in range(1, 4):
            self.view_gp_masks.append(
                torch.from_numpy(np.load('/mnt/data/Yunfei/Study/Baseline_MVDet/Projection_files/'
                                         'Single_view_height_projeciton_relation' +
                                         f'/proj_viewview{view}_mask_outsize{output_size}.npy')).to(
                    self.devicenum))
        # variant_height = 1, take [   0.   583 , 1166, 1750  ] four height level.
        super(SpatialTransformer_v3, self).__init__()

    def __call__(self, inputs, height):
        # for height in self.person_heights:
        for view in range(1, 4):
            # view_input = inputs[view - 1, view, ...]
            # cropped_view_input = view_input
            output_i = self.proj_splat(view, inputs[view - 1:view, ...], height)
            if view == 1:
                output = output_i
            else:
                output = torch.cat([output, output_i], dim=0)
        # output = self.proj_splat(view, inputs, height)

        return output

    def proj_splat(self, view, inputs, height):
        nR = inputs.size()[0]
        fh = inputs.size()[1]
        fw = inputs.size()[2]
        fdim = inputs.size()[3]
        proj_view = self.proj_views_heights[(view, height)].float()
        view_gp_mask = self.view_gp_masks[view - 1].float()
        nV = proj_view.size()[1]
        im_p = proj_view
        im_x, im_y, im_z = im_p[::3, :], im_p[1::3, :], im_p[2::3, :]

        B_im_x = torch.clamp(im_x, 0, fw - 1)
        B_im_y = torch.clamp(im_y, 0, fh - 1)

        B_im_x0 = torch.floor(B_im_x).to(torch.int32).to(self.devicenum)
        B_im_x1 = B_im_x0 + 1
        B_im_x1 = torch.clamp(B_im_x1, 0, fw - 1)

        B_im_y0 = torch.floor(B_im_y).to(torch.int32).to(self.devicenum)
        B_im_y1 = B_im_y0 + 1
        B_im_y1 = torch.clamp(B_im_y1, 0, fh - 1)

        B_im_x0_f, B_im_x1_f = B_im_x0.to(torch.float32), B_im_x1.to(torch.float32)
        B_im_y0_f, B_im_y1_f = B_im_y0.to(torch.float32), B_im_y1.to(torch.float32)

        B_ind_grid = torch.arange(0, nR).to(self.devicenum)
        B_ind_grid = B_ind_grid.unsqueeze_(1)
        B_im_ind = B_ind_grid.repeat(1, nV)

        def _get_gather_inds(x, y):
            temp = torch.reshape(torch.stack([B_im_ind, y, x], dim=2), [-1, 3]).to(self.devicenum)
            return temp

        # Gather  values
        B_Ia = self.gather_nd(inputs, _get_gather_inds(B_im_x0, B_im_y0))
        B_Ib = self.gather_nd(inputs, _get_gather_inds(B_im_x0, B_im_y1))
        B_Ic = self.gather_nd(inputs, _get_gather_inds(B_im_x1, B_im_y0))
        B_Id = self.gather_nd(inputs, _get_gather_inds(B_im_x1, B_im_y1))

        # Calculate bilinear weights
        B_wa = (B_im_x1_f - B_im_x) * (B_im_y1_f - B_im_y)
        # #print("wa.size()",wa.size())
        B_wb = (B_im_x1_f - B_im_x) * (B_im_y - B_im_y0_f)
        B_wc = (B_im_x - B_im_x0_f) * (B_im_y1_f - B_im_y)
        B_wd = (B_im_x - B_im_x0_f) * (B_im_y - B_im_y0_f)
        B_wa = torch.reshape(B_wa, [-1, 1])
        B_wb = torch.reshape(B_wb, [-1, 1])
        B_wc = torch.reshape(B_wc, [-1, 1])
        B_wd = torch.reshape(B_wd, [-1, 1])
        Bilinear_result = (B_wa * B_Ia + B_wb * B_Ib + B_wc * B_Ic + B_wd * B_Id)
        # print('self.view_gp_mask.shape',self.view_gp_mask.shape)

        # self.Ibilin = torch.reshape(Bilinear_result, [1, self.output_size[0], self.output_size[1], fdim])
        # Ibilin = torch.reshape(Bilinear_result, [192, 160, fdim])
        # Ibilin = np.array(Ibilin.detach().cpu())
        # Ibilin = cv2.resize(Ibilin.detach().cpu().numpy(), (self.output_size[1], self.output_size[0]))
        # print('Ibilin shape', Ibilin.shape)
        Ibilin = torch.reshape(Bilinear_result, [1, self.output_size[0], self.output_size[1], fdim]).float()
        # # add a mask:
        # print('self Ibilin shape',self.Ibilin.shape)
        self.Ibilin = torch.mul(Ibilin, view_gp_mask.to(Ibilin.device)).float()
        # # self.Ibilin.append(Bilinear_result)  # no mask needed.

        return self.Ibilin

    def gather_nd(self, x, coords):
        x = x.contiguous().to(self.devicenum)
        inds = coords.type(torch.float32).mv(torch.FloatTensor([0, x.size()[2], 1]).to(self.devicenum)).type(
            torch.int32)
        x_gather = torch.index_select(x.view(-1, x.size()[-1]), 0, inds)
        return x_gather


def get_depth_maps():
    depth_maps_dir = '/home/yunfei/Data/CityStreet/ROI_maps/Distance_maps'
    depth_map_dir_list = sorted(os.listdir(depth_maps_dir))
    depth_map_array_list = []
    for i_dir in depth_map_dir_list:
        a = np.load(os.path.join(depth_maps_dir, i_dir))['arr_0']  # [380,676]
        depth_map_array_list.append(a)
    depth_map = np.stack(depth_map_array_list, axis=0)
    depth_map = torch.from_numpy(depth_map)
    depth_map = torch.sqrt(depth_map)
    depth_map /= 100
    # 标准化
    # mean_x = np.mean(depth_map, axis=0)
    # std_x = np.std(depth_map, axis=0)
    # depth_map = (depth_map - mean_x) / std_x
    # 归一化
    view_size, h, w = list(depth_map.shape)
    # depth_map = torch.from_numpy(depth_map).reshape(view_size, -1)
    # depth_map = torch.nn.functional.normalize(depth_map, p=2, dim=0)
    # depth_map = depth_map.reshape(3, h,w)
    return depth_map


def tensor_visualize(x, image_title='map', isVertical=False):
    pixel_max = x.max()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=pixel_max)
    if isVertical:
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(6, 10))
    else:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))
    fig.suptitle(image_title)
    for i in range(x.shape[0]):
        mappable_i = matplotlib.cm.ScalarMappable(norm=norm)
        im = axs[i].imshow(x[i].detach().cpu().squeeze(), norm=norm)
        # axs[i].set_xticks([])
        # axs[i].set_yticks([])
        axs[i].set_title(f'View_{i + 1}')
    fig.colorbar(mappable_i, ax=axs)
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    import tensorflow as tf

    # from multiview_detector.MultiHeightFusion.processing_layer import camera_sel_fusion_layer_rbm2_v2
    height = 1750
    img_path1 = '/home/yunfei/Data/CityStreet/test_persons/frame_0636_v1.jpg'
    img_path2 = '/home/yunfei/Data/CityStreet/test_persons/frame_0636_v2.jpg'
    img_path3 = '/home/yunfei/Data/CityStreet/test_persons/frame_0636_v3.jpg'
    img_paths = [img_path1, img_path2, img_path3]
    proj_shape = (768, 640)
    # proj_shape = (384, 320)
    # proj_shape = (288, 240)
    # proj_shape = (192, 160)
    # input_shape = (1, 1520, 2704, 3)
    input_shape = (1, 380, 676, 1)
    print('t1:', datetime.datetime.now())
    # depyh_scales = 4
    # depth_map = get_depth_maps()
    # print(depth_map.shape)
    STN = SpatialTransformer_v3(input_shape, proj_shape, 'cpu', person_heights=[height])
    # print(STN.proj_views_heights[(1, 0)].shape)
    # all_depthmaps = []
    #
    # tensor_visualize(depth_map, 'Depth map after sqrt and divided operation', isVertical=True)
    #
    # for i in range(3):
    #     depth_map_projection = STN.proj_splat(view=i + 1, inputs=depth_map[i:i + 1].unsqueeze(3), height=1750)
    #     all_depthmaps.append(depth_map_projection)
    #
    # all_depthmaps = torch.cat(all_depthmaps, 0)
    # # hw_random=[100,100]
    # h = 100
    # w = 100
    # patch_depth_maps = all_depthmaps[:, h:h + 384, w:w + 320, :]
    # tensor_visualize(all_depthmaps, 'all_dmap')
    # tensor_visualize(patch_depth_maps, 'patch_dmap')
    # all_depthmaps = tf.convert_to_tensor(all_depthmaps)
    # pytorch version
    # fusion_world = []
    fig = plt.figure()
    subplt1 = fig.add_subplot(131, title="view1")
    subplt2 = fig.add_subplot(132, title="view2")
    subplt3 = fig.add_subplot(133, title="view3")
    inputs = []
    for view, pathi in enumerate(img_paths):
        img_in = Image.open(pathi).convert('RGB').resize((676, 380))
        # img_i
        img_in = torch.from_numpy(np.asarray(img_in))
        inputs.append(img_in)
        print('img_in shape', img_in.shape)
    inputs = torch.stack(inputs, dim=0)
    outputimg = STN(inputs=inputs, height=height).int()
    res_dir = '/mnt/data/Yunfei/Study/Multi_model_test_results/Img_results'
    pixel_max = outputimg.max()
    pixel_min = outputimg.min()
    norm = matplotlib.colors.Normalize(vmin=pixel_min, vmax=pixel_max)
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
    # fig.suptitle
    for i in range(3):
        mappable_i = matplotlib.cm.ScalarMappable(norm=norm)
        im = axs[i].imshow(outputimg[i].detach().cpu().squeeze(), norm=norm)
        # im = axs[i].imshow(x[i].detach().cpu().squeeze(), norm=norm)
        # axs[i].set_title(f'GT_Under_view{i + 1}')
    mappable_last = matplotlib.cm.ScalarMappable(norm=norm)
    # im_l = axs[3].imshow(y.squeeze().cpu().numpy())
    # axs[3].set_title('GT')
    # fig.colorbar(mappable_i, ax=axs)
    plt.savefig(os.path.join(res_dir, f'heightP{height}_RGB Projection'))
    # plt.show()
    plt.close(fig)
    # print('inputs_shape', inputs.shape)
    # print('outputimg shape', outputimg.shape)
    # # # plt.imshow(img_in)
    # # # plt.show()
    # # img_in = torch.from_numpy(img_in).unsqueeze(0)
    # # world_proj = STN(inputs=img_in, height=0)
    # # # plt.imshow(world_proj.squeeze().cpu().numpy())
    # # # plt.savefig('./treeimgs/view{}_shape={}x{}.png'.format(view, proj_shape[0], proj_shape[1]))
    # # # plt.show()
    # # print(world_proj.shape)
    # # fusion_world.append(world_proj.float())
    # subplt1.imshow(outputimg[0, ...].float().squeeze().cpu().numpy().astype(np.uint8))
    # subplt2.imshow(outputimg[1, ...].float().squeeze().cpu().numpy().astype(np.uint8))
    # subplt3.imshow(outputimg[2, ...].float().squeeze().cpu().numpy().astype(np.uint8))
    # # plt.savefig('./treeimgs/ground_plane_height_test.jpg')
    #
    # plt.show()
    # plt.close(fig)
    #
    # # fusion_world = torch.cat(fusion_world, dim=3).permute(0, 3, 1, 2)
    # # print(fusion_world.shape)
    # # plt.imshow(torch.norm(fusion_world[0].detach(), dim=0).cpu().numpy())
    # # plt.show()
    #
    # print('t3', datetime.datetime.now())


def middle_visualize(x, y, res_dir, title='None'):
    pixel_max = max(x.max(), y.max())
    pixel_min = min(x.min(), y.min())
    norm = matplotlib.colors.Normalize(vmin=pixel_min, vmax=pixel_max)
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(12, 5))
    fig.suptitle(title)
    for i in range(3):
        mappable_i = matplotlib.cm.ScalarMappable(norm=norm)
        im = axs[i].imshow(x[i].detach().cpu().squeeze(), norm=norm)
        # axs[i].set_title(f'GT_Under_view{i + 1}')
    mappable_last = matplotlib.cm.ScalarMappable(norm=norm)
    im_l = axs[3].imshow(y.squeeze().cpu().numpy())
    # axs[3].set_title('GT')
    # fig.colorbar(mappable_i, ax=axs)
    plt.savefig(os.path.join(res_dir, title))
    # plt.show()
    plt.close(fig)
