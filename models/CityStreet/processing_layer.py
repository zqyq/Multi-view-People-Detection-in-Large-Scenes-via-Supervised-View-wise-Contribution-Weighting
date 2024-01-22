import matplotlib
import torch
import torch.nn as nn
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Layer
from utils.person_help import vis

class camera_sel_fusion_layer_rbm2_v2(nn.Module):
    def __init__(self, **kwargs):
        self.patch_num = kwargs['patch_num']
        self.view_size = kwargs['view_size']
        self.batch_size = 1
        # self.out_channel = kwargs['out_channel']
        super(camera_sel_fusion_layer_rbm2_v2, self).__init__()

    def visualize(self, x, title='Mp'):
        x = x.reshape(self.patch_num, self.view_size, x.shape[1], x.shape[2], x.shape[3])
        pixel_max = x.max()
        norm = matplotlib.colors.Normalize(vmin=0, vmax=pixel_max)
        for p in range(self.patch_num):
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
            fig.suptitle(title)
            for i in range(self.view_size):
                mappable_i = matplotlib.cm.ScalarMappable(norm=norm)
                im = axs[i].imshow(x[p][i].detach().cpu().squeeze(), norm=norm)
                # axs[i].set_xticks([])
                # axs[i].set_yticks([])
                axs[i].set_title(f'View_{i + 1}')
            fig.colorbar(mappable_i, ax=axs)
            plt.show()
            plt.close(fig)

    def forward(self, x):
        batch_size = self.batch_size
        view_size = self.view_size
        patch_num = self.patch_num
        # self.visualize(x)
        b_v_p_size = x.shape[0]  # actually, it's b*v*p
        height = x.shape[1]
        width = x.shape[2]
        num_channels = x.shape[3]

        x = torch.reshape(x, (batch_size, view_size, patch_num, height, width, num_channels))
        x = torch.permute(x, (0, 2, 1, 3, 4, 5))
        x = torch.reshape(x, (batch_size * patch_num, view_size, height, width, num_channels))

        x_clip = torch.clamp(x, 0, 1)

        x_clip2 = (1 - x_clip) * 1e8

        x_e8 = x + x_clip2
        x_e8 = torch.log(x_e8)

        x_min = torch.min(x_e8, dim=1, keepdim=True)[0]
        x_min_tile = torch.tile(x_min, (1, view_size, 1, 1, 1))

        # x_sum = torch.reduce_sum(x, axis=1, keep_dims=True)
        x_sum = torch.max(x, dim=1, keepdim=True)[0]
        x_sum_clip = torch.clamp(x_sum, 0, 1)
        x_sum_clip2 = 1 - x_sum_clip

        x_dist = -(torch.square(x_e8 - x_min_tile) / (1))  # x_min_tile*10, 200, 100
        x_dist2 = torch.exp(x_dist)
        x_dist2_mask = torch.multiply(x_dist2, x_clip)

        x_dist2_mask_sum = torch.sum(x_dist2_mask, dim=1, keepdim=True)
        x_dist2_mask_sum2 = torch.tile(x_dist2_mask_sum + x_sum_clip2, (1, view_size, 1, 1, 1))

        x_dist2_mask_sum2_softmax = torch.divide(x_dist2_mask, x_dist2_mask_sum2)
        # x_dist2_mask/x_dist2_mask_sum2
        # torch.nn.softmax(x_dist2_mask_sum2, axis=1)

        x_dist2_mask_sum2_softmax_mask = torch.multiply(x_dist2_mask_sum2_softmax, x_clip)

        x_dist2_mask_sum2_softmax_mask = torch.reshape(x_dist2_mask_sum2_softmax_mask,
                                                       (batch_size, patch_num, view_size, height, width, num_channels))
        x_dist2_mask_sum2_softmax_mask = torch.permute(x_dist2_mask_sum2_softmax_mask, (0, 2, 1, 3, 4, 5))
        x_dist2_mask_sum2_softmax_mask = torch.reshape(x_dist2_mask_sum2_softmax_mask,
                                                       (
                                                           batch_size * view_size * patch_num, height, width,
                                                           num_channels))
        # x_dist2_mask_sum2_softmax_mask = torch.tile(x_dist2_mask_sum2_softmax_mask, [1, 1, 1, 256])

        return x_dist2_mask_sum2_softmax_mask  # output_mask


class camera_sel_fusion_layer_rbm2_full_szie(nn.Module):
    def __init__(self, **kwargs):
        self.view_size = kwargs['view_size']
        self.batch_size = 1
        # self.out_channel = kwargs['out_channel']
        super(camera_sel_fusion_layer_rbm2_full_szie, self).__init__()

    def visualize(self, x, title='Mp'):
        # x = x.reshape(self.view_size, x.shape[1], x.shape[2], x.shape[3])
        pixel_max = x.max()
        norm = matplotlib.colors.Normalize(vmin=0, vmax=pixel_max)
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
        fig.suptitle(title)
        for i in range(self.view_size):
            mappable_i = matplotlib.cm.ScalarMappable(norm=norm)
            im = axs[i].imshow(x[i].detach().cpu().squeeze(), norm=norm)
            # axs[i].set_xticks([])
            # axs[i].set_yticks([])
            axs[i].set_title(f'View_{i + 1}')
        fig.colorbar(mappable_i, ax=axs)
        plt.show()
        plt.close(fig)

    def forward(self, x):
        batch_size = self.batch_size
        view_size = self.view_size

        x_clip = torch.clamp(x, 0, 1)

        x_clip2 = (1 - x_clip) * 1e8

        x_e8 = x + x_clip2
        x_e8 = torch.log(x_e8)

        x_min = torch.min(x_e8, dim=0, keepdim=True)[0]
        x_min_tile = torch.tile(x_min, (view_size, 1, 1, 1))

        x_sum = torch.max(x, dim=0, keepdim=True)[0]
        x_sum_clip = torch.clamp(x_sum, 0, 1)
        x_sum_clip2 = 1 - x_sum_clip

        x_dist = -(torch.square(x_e8 - x_min_tile) / (1))  # x_min_tile*10, 200, 100
        x_dist2 = torch.exp(x_dist)
        x_dist2_mask = torch.multiply(x_dist2, x_clip)

        x_dist2_mask_sum = torch.sum(x_dist2_mask, dim=0, keepdim=True)
        x_dist2_mask_sum2 = torch.tile(x_dist2_mask_sum + x_sum_clip2, (view_size, 1, 1, 1))
        x_dist2_mask_sum2_softmax = torch.divide(x_dist2_mask, x_dist2_mask_sum2)
        x_dist2_mask_sum2_softmax_mask = torch.multiply(x_dist2_mask_sum2_softmax, x_clip)

        return x_dist2_mask_sum2_softmax_mask

class view_pooling_layer(nn.Module):
    def __init__(self, batch_size=1, view_size=3, patch_num=2, **kwargs):
        self.batch_size = batch_size
        self.view_size = view_size
        self.patch_num = patch_num
        super(view_pooling_layer, self).__init__()

    def forward(self, x):
        batch_size = self.batch_size
        view_size = self.view_size
        patch_num = self.patch_num

        feature_view_pooled = torch.reshape(x, (patch_num, view_size,
                                                x.shape[1], x.shape[2], x.shape[3]))
        feature_view_pooled = torch.max(feature_view_pooled, dim=1, keepdim=False)[0]
        # feature_view_pooled = torch.reshape(feature_view_pooled, (batch_size, patch_num,
        #                                                           x.shape[1], x.shape[2], x.shape[3]))
        return feature_view_pooled

class view_pooling_layer_size(nn.Module):
    def __init__(self, batch_size=1, view_size=3, **kwargs):
        self.batch_size = batch_size
        self.view_size = view_size
        super(view_pooling_layer_size, self).__init__()

    def forward(self, x):
        batch_size = self.batch_size
        view_size = self.view_size

        feature_view_pooled = torch.max(x, dim=1, keepdim=False)[0]
        # feature_view_pooled = torch.reshape(feature_view_pooled, (batch_size, patch_num,
        #                                                           x.shape[1], x.shape[2], x.shape[3]))
        return feature_view_pooled

class Dmap_consist_loss_layer(nn.Module):
    def __init__(self,
                 **kwargs):
        super(Dmap_consist_loss_layer, self).__init__()

    def forward(self, x):
        x_output = x[0]
        view_gp_output = x[1]
        output_loss = x_output - view_gp_output
        return output_loss


if __name__ == '__main__':
    from utils.person_help import vis
    # x = tf.random.normal([3, 10, 10, 1], dtype=tf.float32, seed=1)
    x=torch.rand(3,10,10,1)
    tf_camera_sel_model = camera_sel_fusion_layer_rbm2_full_szie(batch_size=1, view_size=3)
    out = tf_camera_sel_model(x)
    print(out)
