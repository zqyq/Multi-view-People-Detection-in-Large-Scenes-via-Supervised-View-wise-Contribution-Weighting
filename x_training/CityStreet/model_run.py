from multiview_detector.utils.logger import Logger
# from utils.draw_curve import draw_curve
import torch
import numpy as np
import torchvision.transforms as T
# from torchvision.utils import save_image
import argparse
import sys
import shutil
from distutils.dir_util import copy_tree
import datetime
import os
from multiview_detector.utils.image_utils import img_color_denormalize

from multiview_detector.models.CityStreet.CityStreet_Detector import PerspTransDetector
from multiview_detector.trainer.CityStreet.CityStreet_Trainer import PerspectiveTrainer
# from torch.utils.tensorboard import SummaryWriter
from multiview_detector.utils.load_model import loadModel
import tqdm
from multiview_detector.datasets.CityStreet.Citystreet import Citystreet as Base
from multiview_detector.datasets.CityStreet.frame_CityStreet import frameDataset


# from torch.utils.tensorboard import SummaryWriter


def model_run(args):
    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = True

    # dataset
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_trans = T.Compose([T.Resize([760, 1352]), T.ToTensor(), normalize, ])
    # train_trans = T.Compose([T.ToTensor(), normalize])
    # dataset root
    data_root = '/mnt/data/Datasets/CityStreet'  # Self-defined dataset direction
    base = Base(data_root)
    train_set = frameDataset(args, base, train=True, _transform=train_trans)
    val_set = frameDataset(args, base, train=False, _transform=train_trans)
    # dataloader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False,
                                             num_workers=1, pin_memory=False)
    print(f'Train set num:{len(train_set)}, Validation set num:{len(val_set)}')
    # logging
    logdir = f'logs/{args.dataset}_dataset/{args.arch}/{args.variant}/' + datetime.datetime.today().strftime(
        '%Y-%m-%d_%H-%M-%S') + f'mo{args.momentum}_wd{args.weight_decay}_lr{args.lr}_' \
                               f'freeze_2D{args.fix_2D}_w{args.weight_2D}_svp{args.fix_svp}_w{args.weight_svp}_' \
                               f'lrs{args.lr_scheduler}_epo{args.epochs}_valEpo{args.val_epochs}_' \
                               f'ct{args.cls_thres}_nt{args.nms_thres}_dt{args.dist_thres}'

    # Save some important .py files
    ##################################
    scripts_dir = os.path.join(logdir, 'scripts')
    # os.makedirs(scripts_dir,exist_ok=True)
    copy_tree('/mnt/data/Yunfei/Study/MVD_VCW/multiview_detector/datasets/CityStreet', scripts_dir)
    copy_tree('/mnt/data/Yunfei/Study/MVD_VCW/multiview_detector/models/CityStreet', scripts_dir)
    copy_tree('/mnt/data/Yunfei/Study/MVD_VCW/multiview_detector/trainer/CityStreet', scripts_dir)
    copy_tree('/mnt/data/Yunfei/Study/MVD_VCW/multiview_detector/x_training/CityStreet', scripts_dir)
    ##################################

    sys.stdout = Logger(os.path.join(logdir, 'log.txt.txt'), )
    print('Settings:')
    print(vars(args))
    # args.pretrain = '/mnt/data/Yunfei/Study/MVD_VCW/logs/citystreet_dataset/vgg16/2D_SVP_VCW/2023-12-19_10-22-44mo0.9_wd0' \
                    # '.0001_lr0.001_freeze_2D1_w1_svp0_w10_lrsonecycle_epo50_valEpo5_ct0.4_nt40_dt80/latest_2D_SVP_VCW_model.pth'
    # model
    model = PerspTransDetector(train_set, args,logdir)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.lr_scheduler == 'lambda':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=lambda epoch: 1 / (
                                                              1 + args.lr_decay * epoch) ** epoch)
    elif args.lr_scheduler == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                                        epochs=args.epochs)
    elif args.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.6)
        args.epochs = 50
    else:
        raise Exception('Must choose from [lambda, onecycle]')

    if args.pretrain is not None:
        print(f'Loading the weight of a pretrained model from {args.pretrain}.')
        model = loadModel(model, args.pretrain, 'model')

    # Trainer
    trainer = PerspectiveTrainer(model, args, logdir, denormalize)
    # training and validation
    trainer.val(args.variant, val_loader, os.path.join(logdir, 'test.txt'), epoch=0)
    for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
        trainer.train(args.variant, train_loader, epoch, optimizer, scheduler)
        trainer.val(args.variant, val_loader, os.path.join(logdir, 'test.txt'), epoch)
    