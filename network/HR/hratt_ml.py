import torch.nn as nn
import torch
from .modules import DWT,IWT
from torch.nn import functional as F
import pdb
'''
对应的训练文件夹是hratt_ml
'''
def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                     groups=groups)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        # import pdb
        # pdb.set_trace()
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y



class Net(nn.Module):
    def __init__(self, in_channels=12, out_channels=12):
        super(Net, self).__init__()
        pass

    def forward(self, x):
        _, _, pad_h, pad_w = x.size()
        if pad_h % 2 != 0 or pad_w % 2 != 0:
            h_pad_len = 2 - pad_h % 2
            w_pad_len = 2 - pad_w % 2
            x = F.pad(x, (0, w_pad_len, 0, h_pad_len), mode='reflect')

        out = x
        return out[:,:,:pad_h,:pad_w]

