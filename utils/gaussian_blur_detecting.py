import matplotlib.pyplot as plt
import numpy as np


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


if __name__ == '__main__':
    center = [100, 100]
    heatmap = np.zeros((384, 320))
    map_sigma = 3
    # gaussian_kernel_sum = gaussian2D(shape=[6 * map_sigma + 1, 6 * map_sigma + 1], sigma=map_sigma).sum()
    # print(gaussian_kernel_sum)
    for cx in [100, 95]:
        for cy in [80]:
            center = (cx, cy)
            heatmap = draw_umich_gaussian(heatmap, center, 3, 1)
            plt.imshow(heatmap)
            plt.show()
    print(heatmap.sum())
