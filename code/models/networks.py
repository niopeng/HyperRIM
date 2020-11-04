import functools
import torch
import torch.nn as nn
from torch.nn import init
import models.modules.architecture as arch


####################
# initialize
####################
def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    print('initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################
def define_G(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'IMRRDB_net':
        netG = arch.IMRRDBNet(in_nc=opt_net['in_nc'], code_nc=opt_net['code_nc'], out_nc=opt_net['out_nc'],
                              num_residual_channels=opt_net['num_residual_channels'],
                              num_dense_channels=opt_net['num_dense_channels'], num_blocks=opt_net['num_blocks'],
                              upscale=opt['scale'], act_type='leakyrelu', upsample_kernel_mode="nearest",
                              feat_scales=opt_net['feat_scales'], map_nc=opt_net['map_nc'],
                              latent_nc=opt_net['latent_nc'], no_upsample=opt_net['no_upsample'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    if opt['is_train']:
        scale = 0.1 if 'init_scale' not in opt_net else opt_net['init_scale']
        init_type = 'kaiming' if 'init_type' not in opt_net else opt_net['init_type']
        init_weights(netG, init_type=init_type, scale=scale)
    if gpu_ids:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG, device_ids=gpu_ids) if len(gpu_ids) > 1 else nn.DataParallel(netG)
    return netG


def define_F(opt):
    gpu_ids = opt['gpu_ids']
    netF = arch.LPNet()
    netF.eval()
    if gpu_ids:
        assert torch.cuda.is_available()
        netF = nn.DataParallel(netF, device_ids=gpu_ids) if len(gpu_ids) > 1 else nn.DataParallel(netF)
    return netF
