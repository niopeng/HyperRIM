from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

####################
# Basic blocks
####################


def act(act_type, inplace=True, neg_slope=0.2):
    act_type = act_type.lower()
    if act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'tanh':
        layer = nn.Tanh()
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True) + eps)
    return in_feat / (norm_factor + eps)


class ShortcutBlock(nn.Module):
    # Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x, extra_x=None):
        if extra_x is None:
            output = x + self.sub(x)
        else:
            output = x + self.sub(x, extra_x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size, dilation=1, bias=True, pad_type='zero', act_type='leakyrelu'):
    '''
    Conv layer with weight normalization (NIPS 2016), activation
    '''
    padding = get_valid_padding(kernel_size, dilation)
    padding = padding if pad_type == 'zero' else 0

    c = weight_norm(nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=bias),
                    name='weight')
    a = act(act_type) if act_type else None
    return sequential(c, a)


def upconv_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, bias=True, pad_type='zero', act_type='leakyrelu',
                 mode='nearest'):
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, bias=bias, pad_type=pad_type, act_type=act_type)
    return sequential(upsample, conv)


class ResidualDenseBlock(nn.Module):
    '''
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, bias=True, pad_type='zero', act_type='leakyrelu'):
        super(ResidualDenseBlock, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(nc, gc, kernel_size, bias=bias, pad_type=pad_type, act_type=act_type)
        self.conv2 = conv_block(nc+gc, gc, kernel_size, bias=bias, pad_type=pad_type, act_type=act_type)
        self.conv3 = conv_block(nc+2*gc, gc, kernel_size, bias=bias, pad_type=pad_type, act_type=act_type)
        self.conv4 = conv_block(nc+3*gc, gc, kernel_size, bias=bias, pad_type=pad_type, act_type=act_type)
        self.conv5 = conv_block(nc+4*gc, nc, 3, bias=bias, pad_type=pad_type, act_type=None)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x


class ScalingLayer(nn.Module):
    # For rescaling the input to vgg16
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class RerangeLayer(nn.Module):
    # Change the input from range [-1., 1.] to [0., 1.]
    def __init__(self):
        super(RerangeLayer, self).__init__()

    def forward(self, inp):
        return (inp + 1.) / 2.


class NetLinLayer(nn.Module):
    ''' A single linear layer used as placeholder for LPIPS learnt weights '''
    def __init__(self):
        super(NetLinLayer, self).__init__()
        self.weight = None

    def forward(self, inp):
        out = self.weight * inp
        return out


class RRDB(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, bias=True, pad_type='zero', act_type='leakyrelu'):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock(nc, kernel_size, gc, bias, pad_type, act_type)
        self.RDB2 = ResidualDenseBlock(nc, kernel_size, gc, bias, pad_type, act_type)
        self.RDB3 = ResidualDenseBlock(nc, kernel_size, gc, bias, pad_type, act_type)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x


class StyleBlock(nn.Module):
    '''
    Style Block: Rescale each RRDB output
    '''

    def __init__(self, rrdbs, transformations, lr_conv):
        super(StyleBlock, self).__init__()
        assert len(rrdbs) == len(transformations)
        self.nb = len(rrdbs)
        for i, rrdb in enumerate(rrdbs):
            self.add_module("%d" % i, rrdb)
            self.add_module("transform_%d" % i, transformations[i])
        self.lr_conv = lr_conv

    def forward(self, x, x_feat):
        for i in range(self.nb):
            rrdb_out = getattr(self, "%d" % i)(x)
            tran_out = getattr(self, "transform_%d" % i)(x_feat)
            bs, nc, w, h = rrdb_out.shape
            norm_layer = nn.InstanceNorm2d(nc, affine=False)
            x = (1. + tran_out[:, :nc].reshape(bs, nc, 1, 1)).expand(bs, nc, w, h) * norm_layer(rrdb_out) + \
                tran_out[:, nc:].reshape(bs, nc, 1, 1).expand(bs, nc, w, h)
        out = self.lr_conv(x)
        return out
