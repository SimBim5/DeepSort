from __future__ import absolute_import

import torch
import math
from torch import nn
from torch.nn import functional as F


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(self.in_channels, self.inter_channels,
                         kernel_size=1, stride=1, padding=0, bias=True)
        self.theta = conv_nd(self.in_channels, self.inter_channels,
                             kernel_size=1, stride=1, padding=0, bias=True)
        self.phi = conv_nd(self.in_channels, self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        if sub_sample:
            if dimension == 3:
                self.g = nn.Sequential(self.g, max_pool((1,2,2)))
                self.phi = nn.Sequential(self.phi, max_pool((1,2,2)))
            else:
                self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(self.inter_channels, self.in_channels,
                        kernel_size=1, stride=1, padding=0, bias=True),
                bn(self.in_channels)
            )
        else:
            self.W = conv_nd(self.inter_channels, self.in_channels,
                             kernel_size=1, stride=1, padding=0, bias=True)
        
        # init
        for m in self.modules():
            if isinstance(m, conv_nd):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, bn):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if bn_layer:
            nn.init.constant_(self.W[1].weight.data, 0.0)
            nn.init.constant_(self.W[1].bias.data, 0.0)
        else:
            nn.init.constant_(self.W.weight.data, 0.0)
            nn.init.constant_(self.W.bias.data, 0.0)


    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f = F.softmax(f, dim=-1)

        y = torch.matmul(f, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        y = self.W(y)
        z = y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


if __name__ == '__main__':
    from torch.autograd import Variable
    import torch
    sub_sample = False

    img = Variable(torch.zeros(2, 4, 5))
    net = NONLocalBlock1D(4, sub_sample=sub_sample, bn_layer=False)
    out = net(img)
    print(out.size())

    img = Variable(torch.zeros(2, 3, 256, 128))
    net = NONLocalBlock2D(3, sub_sample=sub_sample)
    out = net(img)
    print(out.size())

    img = Variable(torch.zeros(2, 3, 8, 256, 128))
    net = NONLocalBlock3D(3, sub_sample=sub_sample)
    out = net(img)
    print(out.size())