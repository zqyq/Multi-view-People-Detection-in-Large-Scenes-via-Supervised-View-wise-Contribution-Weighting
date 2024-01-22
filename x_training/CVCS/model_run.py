from utils.logger import Logger
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
from utils.image_utils import img_color_denormalize
from models.CVCS.CVCS_Detector import PerspTransDetector
from trainer.CVCS.CVCS_Trainer import PerspectiveTrainer
# from torch.utils.tensorboard import SummaryWriter
from utils.load_model import loadModel
import tqdm
from datasets.CVCS.CVCS import CVCS as Base
from datasets.CVCS.frame_CVCS import frameDataset


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
    # train_trans = T.Compose([T.Resize([720, 1280]), T.ToTensor(), normalize, ])
    train_trans = T.Compose([T.ToTensor(), normalize])
    # dataset root
    data_root = '/mnt/data/Datasets/CVCS'
    base = Base(data_root)
    train_set = frameDataset(base, train=True, _transform=train_trans)
    val_set = frameDataset(base, train=False, _transform=train_trans)
    # dataloader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False,
                                             num_workers=1, pin_memory=False)
    print(f'Trainset num:{len(train_set)}, Validationset num:{len(val_set)}')
    # logging
    logdir = f'logs/{args.dataset}_dataset/{args.arch}/{args.variant}/' + datetime.datetime.today().strftime(
        '%Y-%m-%d_%H-%M-%S') + f'bs_{args.batch_size}_mo{args.momentum}_wd{args.weight_decay}_lr{args.lr}_' \
                               f'lrs{args.lr_scheduler}_epo{args.epochs}_valEpo{args.val_epochs}_' \
                               f'ct{args.cls_thres}_nt{args.nms_thres}_dt{args.dist_thres}'
    # Save some important .py files
    ##################################
    scripts_dir = os.path.join(logdir, 'scripts')
    copy_tree(os.path.join(args.proj_root,'datasets/CVCS'), scripts_dir)
    copy_tree(os.path.join(args.proj_root,'models/CVCS'), scripts_dir)
    copy_tree(os.path.join(args.proj_root,'trainer/CVCS'), scripts_dir)
    copy_tree(os.path.join(args.proj_root,'x_training/CVCS'), scripts_dir)
    ##################################
    sys.stdout = Logger(os.path.join(logdir, 'log.txt.txt'), )
    print('Settings:')
    print(vars(args))
    # writer = SummaryWriter(os.path.join(logdir, 'tensorboard'))
    # pretrainer model direction
    # args.pretrain = '/mnt/data/Yunfei/Study/MVD_VCW/logs/cvcs_dataset/resnet18/2D_SVP_VCW/2023-12-11_19-27-11bs_1' \
    #                 '_mo0.9_wd0.0001_lr0.01_lrsonecycle_epo50_valEpo5_ct0.4_nt5_dt5/latest_2D_SVP_VCW_model.pth'
    # model
    model = PerspTransDetector(train_set, args)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.lr_scheduler == 'lambda':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=lambda epoch: 1 / (
                                                              1 + args.lr_decay * epoch) ** epoch)
    elif args.lr_scheduler == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                                        epochs=args.epochs)
    else:
        raise Exception('Must choose from [lambda, onecycle]')

    if args.pretrain is not None:
        print(f'Loading the weight of a pretrained model from {args.pretrain}.')
        model = loadModel(model,args.pretrain)  # 之后键值改为’model‘

    # Trainer
    trainer = PerspectiveTrainer(model, args, logdir, denormalize, batch_size=train_set.batch_size,
                                 view_size=train_set.view_size, patch_num=train_set.patch_num)
    # training and validation
    # trainer.val(args.variant, val_loader, os.path.join(logdir, 'test.txt'), epoch=0)
    for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
        trainer.train(args.variant, train_loader, epoch, optimizer, scheduler)
        if epoch % args.val_epochs == 0:
            trainer.val(args.variant, val_loader, os.path.join(logdir, 'test.txt'), epoch)
