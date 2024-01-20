import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.init as init


def purevis(x):
    color_map = 'gray_r'
    plt.imshow(x.squeeze().numpy(), cmap=color_map)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def savefeatvis(fmap, save):
    fmap = fmap.detach().squeeze()
    plt.imshow(torch.norm(fmap, dim=0).cpu().numpy())
    plt.savefig(save)


def vis(*args, ticks=True, colorbar=False, coshow=False):
    for x in args:
        x = x.detach().cpu().squeeze()
        if coshow:
            if x.ndim >= 3:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                # n = x.shape[0]

                im1 = axs[0].imshow(x[0], cmap='viridis', interpolation='nearest')
                axs[0].set_title('view1')
                axs[0].axis('off')

                im2 = axs[1].imshow(x[1], cmap='viridis', interpolation='nearest')
                axs[1].set_title('view2')
                axs[1].axis('off')

                im3 = axs[2].imshow(x[2], cmap='viridis', interpolation='nearest')
                axs[2].set_title('view3')
                axs[2].axis('off')
                if colorbar:
                    cbar = fig.colorbar(im3, ax=axs, orientation='vertical')
                    cbar.set_label('Value')
                plt.subplots_adjust(right=0.85)
                plt.show()
        else:
            if x.ndim >= 3:
                for i in range(x.shape[0]):
                    plt.imshow(x[i].detach().cpu().squeeze())
                    if colorbar:
                        plt.colorbar()
                    if ticks:
                        plt.show()
                    else:
                        plt.xticks([])
                        plt.yticks([])
                        plt.show()
            else:
                plt.imshow(x.detach().cpu().squeeze())
                if colorbar:
                    plt.colorbar()
                if ticks:
                    plt.show()
                else:
                    plt.xticks([])
                    plt.yticks([])
                    plt.show()


def Initialize_net(net, mode='equal'):
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            if mode == 'equal':
                init.constant_(layer.weight, 1 / (layer.out_channels))
            else:
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.Linear):
            init.xavier_uniform_(layer.weight)
            init.constant_(layer.bias, 0.1)
