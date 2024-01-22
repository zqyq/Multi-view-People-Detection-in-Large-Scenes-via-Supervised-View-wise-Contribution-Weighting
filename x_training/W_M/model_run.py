from utils.logger import Logger
# from utils.draw_curve import draw_curve
import torch
import numpy as np
import torchvision.transforms as T
# from torchvision.utils import save_image
import sys
from distutils.dir_util import copy_tree
import datetime
import os
from utils.image_utils import img_color_denormalize
from models.W_M.WM_Detector import PerspTransDetector
from trainer.W_M.WM_trainer import PerspectiveTrainer
# from torch.utils.tensorboard import SummaryWriter
from utils.load_model import loadModel
from datasets.W_M.Wildtrack import Wildtrack
from datasets.W_M.MultiviewX import MultiviewX
from datasets.W_M.frameDataset_head import frameDataset


# from torch.utils.tensorboard import SummaryWriter

def model_run(args):
    # Judge the pretrained model
    if args.pretrain is None:
        raise Exception('There is no pretrained model to test.')
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
    train_trans = T.Compose([T.Resize([720, 1280]), T.ToTensor(), normalize, ])
    # dataset root
    if args.dataset == 'wildtrack':
        base = Wildtrack(os.path.join(args.data_root, 'Wildtrack'))
    elif args.dataset == 'multiviewx':
        base = MultiviewX(os.path.join(args.data_root, 'MultiviewX'))
    else:
        raise Exception('The dataset should be in ["wildtrack", "multiviewx"]')
    # train_set = frameDataset(base, train=True, _transform=train_trans)
    test_set = frameDataset(base, train=False, _transform=train_trans, facofmaxgt=args.facofmaxgt)
    # dataloader
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
    #                                            num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,
                                              num_workers=1, pin_memory=False)
    print(f' test set num:{len(test_set)}')
    # logging
    logdir = f'logs/{args.dataset}_dataset/{args.arch}/test/' + datetime.datetime.today().strftime(
        '%Y-%m-%d_%H-%M-%S') + f'ct{args.cls_thres}_nt{args.nms_thres}_dt{args.dist_thres}'
    # Save some important .py files
    ##################################
    scripts_dir = os.path.join(logdir, 'scripts')
    copy_tree(os.path.join(args.project_root_path, 'utils'), scripts_dir + '/utils')
    copy_tree(os.path.join(args.project_root_path, 'evaluation'), scripts_dir + '/evaluation')
    copy_tree(os.path.join(args.project_root_path, 'x_training/W_M'),
              scripts_dir + '/x_training/W_M')
    copy_tree(os.path.join(args.project_root_path, 'trainer/W_M'), scripts_dir + '/trainer/W_M')
    copy_tree(os.path.join(args.project_root_path, 'models/W_M'), scripts_dir + '/models/W_M')
    copy_tree(os.path.join(args.project_root_path, 'datasets/W_M'), scripts_dir + '/datasets/W_M')
    ##################################

    sys.stdout = Logger(os.path.join(logdir, 'log.txt.txt'), )
    print('Settings:')
    print(vars(args))
    # writer = SummaryWriter(os.path.join(logdir, 'tensorboard'))
    # pretrainer model direction
    # args.pretrain = '/mnt/data/Yunfei/Study/MVD_VCW/logs/cvcs_dataset/resnet18/2D_SVP_VCW/2023-12-11_19-27-11bs_1' \
    #                 '_mo0.9_wd0.0001_lr0.01_lrsonecycle_epo50_valEpo5_ct0.4_nt5_dt5/latest_2D_SVP_VCW_model.pth'
    # model
    model = PerspTransDetector(test_set, args)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # if args.lr_scheduler == 'lambda':
    #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
    #                                                   lr_lambda=lambda epoch: 1 / (
    #                                                           1 + args.lr_decay * epoch) ** epoch)
    # elif args.lr_scheduler == 'onecycle':
    #     scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
    #                                                     epochs=args.epochs)
    # else:
    #     raise Exception('Must choose from [lambda, onecycle]')

    if args.pretrain is not None:
        print(f'Loading the weight of a pretrained model from {args.pretrain}.')
        model = loadModel(model, args.pretrain)  # 之后键值改为’model‘

    # Trainer
    trainer = PerspectiveTrainer(model, args, logdir, denormalize)
    # training and validation
    print('Directly test on the dataset based on the model trained on large dataset.')
    trainer.test(test_loader, os.path.join(logdir, 'test.txt'))
