import torch
from collections import OrderedDict


def loadModel(model, pretrained_dir=None, key='model', optimizer=None, scheduler=None):
    # model
    pretrained_model = torch.load(pretrained_dir)
    model_dict = model.state_dict()
    if key is not None:
        pretrained_dict = {k: v for k, v in pretrained_model[key].items() if k in model_dict}
    else:
        pretrained_dict = pretrained_model
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # optimizer
    if optimizer is not None and 'optimizer' in pretrained_model.keys():
        optimizer.load_state_dict(pretrained_model['optimizer'])
        # scheduler
        if scheduler is not None and 'scheduler' in pretrained_model.keys():
            scheduler.load_state_dict(pretrained_model['scheduler'])
        return model, optimizer, scheduler
    else:
        return model


def Amend_model_keys(pretrained_dir=None, model_key='net', amendKeys=[], save_dir='model.pth'):
    # model
    pretrained_model = torch.load(pretrained_dir)
    new_model_dict = OrderedDict()
    # pretrained_dict = {k: v for k, v in pretrained_model[model_key].items() if k in model_dict}
    K1, K2 = amendKeys
    for k, v in pretrained_model.items():
        if K1 in k:
            newk = K2 + k[len(K1):]
            new_model_dict[newk] = v
        else:
            new_model_dict[k] = v

    torch.save(new_model_dict, save_dir)


if __name__ == '__main__':
    mdir = '/mnt/data/Yunfei/Study/MVD_VCW/logs/citystreet_dataset/vgg16/2D_SVP/model_2D_SVP.pth'
    save_dir = '/mnt/data/Yunfei/Study/MVD_VCW/logs/citystreet_dataset/vgg16/2D_SVP/model_2D_SVP.pth'
    Amend_model_keys(pretrained_dir=mdir, amendKeys=['base_pt', 'base'], save_dir=save_dir)
