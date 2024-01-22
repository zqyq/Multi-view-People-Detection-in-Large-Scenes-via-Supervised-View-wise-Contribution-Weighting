import json
import os

from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils.image_utils import add_heatmap_to_image
from utils.gaussian_blur_detecting import draw_umich_gaussian
from utils.image_utils import img_color_denormalize

denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
img_reduce = 4
sigma = 3
visualize = True
with open('/home/yunfei/Data/CityStreet/labels/via_region_data_view1.json') as file_v1:
    data = json.load(file_v1)
    for frame in data.keys():
        people = data[frame]['regions']
        peopleNum = len(people)
        h, w = 1520 // img_reduce, 2704 // img_reduce
        img_gt = np.zeros(shape=(h, w))
        for i in people:
            cx = int(people[i]['shape_attributes']['cx']) // img_reduce
            cy = int(people[i]['shape_attributes']['cy']) // img_reduce
            if cy < 0 or cy >= h or cx < 0 or cx >= w:
                continue
            img_gt[cy][cx] = 1
            draw_umich_gaussian(img_gt, center=(cx, cy), sigma=sigma, k=1)
        if visualize:
            plt.imshow(img_gt)
            plt.show()
            heatmap0_head = img_gt
            fpath = f'/home/yunfei/Data/CityStreet/image_frames/camera1/{frame}'
            img0 = Image.open(fpath).convert('RGB')
            head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
            # head_cam_result.save(os.path.join(, f'frame{frame}_cam{view}.jpg'))
