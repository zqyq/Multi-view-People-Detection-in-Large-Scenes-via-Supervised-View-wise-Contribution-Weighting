import argparse
import os
import sys

import numpy as np
import torch


# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
def main(args):
    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = True

    # dataset selection
    if args.dataset == 'wildtrack' or args.dataset == 'multiviewx':
        if args.variant == '2D':
            from multiview_detector.x_training.W_M.model_run_2D import model_run
        elif args.variant == '2D_SVP':
            from multiview_detector.x_training.W_M.model_run_2D_SVP import model_run
        elif args.variant == '2D_SVP_3D':
            from multiview_detector.x_training.W_M.model_run_2D_SVP_3D import model_run
        else:
            raise Exception('Wrong variants.')
    elif args.dataset == 'citystreet':
        if args.variant == '2D':
            from multiview_detector.x_training.CityStreet.model_run_2D import model_run
        elif args.variant == '2D_SVP':
            from multiview_detector.x_training.CityStreet.model_run_2D_SVP import model_run
        elif args.variant == '2D_SVP_WS':
            from multiview_detector.x_training.CityStreet.model_run_2D_SVP_WS import model_run
        elif args.variant == '2D_SVP_VCW':
            from multiview_detector.x_training.CityStreet.model_run_2D_SVP_VCW import model_run
        elif args.variant == 'MH_2D_SVP':
            from multiview_detector.x_training.CityStreet.model_run_2D_SVP_Multi_height_MVDet import model_run
        elif args.variant == 'MH_2D_SVP_VCW':
            from multiview_detector.x_training.CityStreet.model_run_2D_SVP_VCW_Multi_height import model_run
        else:
            raise Exception('Wrong variants.')
    elif args.dataset == 'cvcs':
        from multiview_detector.x_training.CVCS.model_run import model_run
    else:
        raise Exception('Input wrong dataset name.')
    args.person_heights = list(map(lambda x: int(x), args.person_heights.split(',')))

    model_run(args)


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--variant', type=str, default='2D_SVP_VCW',
                        choices=['default', '2D', '2D_SVP', '2D_3D', '2D_SVP_3D', '2D_SVP_WS', '2D_SVP_VCW',
                                 'MH_2D_SVP', 'MH_2D_SVP_VCW'])
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'resnet18'])
    parser.add_argument('-d', '--dataset', type=str, default='citystreet',
                        choices=['wildtrack', 'multiviewx', 'citystreet', 'cvcs'])

    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')

    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--lrfac', type=float, default=0.01, help='generating smaller lr')
    parser.add_argument('--lr_decay', '--ld', type=float, default=1e-4)
    parser.add_argument('--lr_scheduler', '--lrs', type=str, default='onecycle', choices=['onecycle', 'lambda', 'step'])
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: None)')
    parser.add_argument('--test', type=str, default=None)

    parser.add_argument('--facofmaxgt', '--fm', type=float, default=1000)
    parser.add_argument('--facofmaxgt_gp', '--fmg', type=float, default=10)

    parser.add_argument('--fix_2D', type=float, default=1)
    parser.add_argument('--fix_svp', type=float, default=0)
    parser.add_argument('--fix_weight', type=float, default=0)

    parser.add_argument('--map_sigma', '--ms', type=float, default=3)
    parser.add_argument('--img_reduce', '--ir', type=float, default=2)
    parser.add_argument('--world_reduce', '--wr', type=float, default=2)

    parser.add_argument('--weight_svp', type=float, default=1)
    parser.add_argument('--weight_2D', type=float, default=1)
    parser.add_argument('--weight_ssv', type=float, default=1)

    parser.add_argument('--cls_thres', '--ct', type=float, default=0.4)
    parser.add_argument('--nms_thres', '--nt', type=float, default=10)
    parser.add_argument('--dist_thres', '--dt', type=float, default=20)

    # parser.add_argument('--person_heights', '--ph', type=list, default=[0, 600, 1200, 1800])
    parser.add_argument('--person_heights', '--ph', type=str, default='1750')
    parser.add_argument('--multiheight', '--mh', type=str, default='3drom', choices=['shot', '3drom'])
    args = parser.parse_args()

    main(args)
