import random
import math

import PIL.Image
import matplotlib.pyplot as plt
import numpy
import numpy as np
import cv2
from PIL import Image
import torch


def random_affine(img, img_gt, hflip=0.5, degrees=(-0, 0), translate=(.2, .2), scale=(0.6, 1.4),
                  shear=(-0, 0),
                  borderValue=(128, 128, 128)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # random.seed(seed)
    border = 0  # width of added border (optional)
    img = np.asarray(img)
    img_gt = np.expand_dims(img_gt, axis=2)

    height = img.shape[0]
    width = img.shape[1]

    # flipping
    F = np.eye(3)
    hflip = np.random.rand() < hflip
    if hflip:
        F[0, 0] = -1
        F[0, 2] = width

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(width / 2, height / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * width + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * height + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R @ F  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue
    # zoom_mat = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 1])
    imw_gt = cv2.warpPerspective(img_gt, M, dsize=(img_gt.shape[1], img_gt.shape[0]), flags=cv2.INTER_LINEAR,
                                 borderValue=0)
    return imw, imw_gt, M


def inverse_random_affine(img, M, borderValue=(128, 128, 128)):
    inverse_M = np.linalg.inv(M)
    height = img.shape[0]
    width = img.shape[1]
    imw = cv2.warpPerspective(img, inverse_M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue
    return imw


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, sigma, k=1):
    radius = int(3 * sigma)
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=sigma)

    x, y = int(center[0]), int(center[1])

    H, W = heatmap.shape

    left, right = min(x, radius), min(W - x, radius + 1)
    top, bottom = min(y, radius), min(H - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


class img_color_denormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean).view([1, -1, 1, 1])
        self.std = torch.FloatTensor(std).view([1, -1, 1, 1])

    def __call__(self, tensor):
        return tensor * self.std.to(tensor.device) + self.mean.to(tensor.device)


def add_heatmap_to_image(heatmap, image):
    heatmap = cv2.resize(np.array(array2heatmap(heatmap)), (image.size))
    cam_result = np.uint8(heatmap * 0.5 + np.array(image) * 0.5)
    cam_result = Image.fromarray(cam_result)
    return cam_result


def array2heatmap(heatmap):
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-8)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_SUMMER)
    heatmap = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    return heatmap


def img_visualize(img):
    if isinstance(img, numpy.ndarray):
        plt.imshow(img)
        plt.show()
    elif isinstance(img, PIL.Image.Image):
        img.show()
    elif isinstance(img, torch.Tensor):
        img = img.detach().cpu().squeeze()
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.permute(1, 2, 0)
            plt.imshow(img)
            plt.show()
        else:
            plt.imshow(torch.norm(img, dim=0))
            plt.show()
    else:
        raise Exception(f'{str(img)} cannot show. ')


def randomRectangle(img, img_gt):
    x, w = np.random.randint(0, img.shape[2], size=(2,))
    y, h = np.random.randint(0, img.shape[1], size=(2,))
    img_reduce = img.shape[1] // img_gt.shape[0]
    # 在图像上绘制矩形
    img[:, y:y + h, x:x + w] = 0
    # 缩小gt
    mask = np.zeros(img_gt.shape)
    x, y, w, h = list(map(lambda i: int(i // img_reduce), [x, y, w, h]))
    img_gt[y:y + h, x:x + w] = 0
    mask[y:y + h, x:x + w] = 1
    return img, img_gt, mask


def vis(img):
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    from scipy.ndimage import gaussian_filter
    import cv2
    import numpy as np

    # 创建一个黑色图像
    img = np.zeros((1520, 2704, 3), np.uint8)
    randomRectangle(img)
