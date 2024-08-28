import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Softmax


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CC_module(nn.Module):

    def __init__(self, in_dim):
        super(CC_module, self).__init__()

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()

        """Q"""
        proj_query = self.query_conv(x)  # (b,c2,h,w)
        # (b*w, h, c2) (b*h, w, c2)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        """K"""
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        """V"""
        proj_value = self.value_conv(x)
        # (b*w, c2, h)(b*h, c2, h)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        # (b*w, h, c2) * (b*w, h, c2)->(b*w, h, h)->(b,h, w, h)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).\
            view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        # (b*h, w, c2)*(b*h, w, c2)->(b*h, w, w)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        # (b,h, w, w) (b,h, w, h) -> (b,h, w, h+w)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        # (b*w,h,h) (b*h,w,w)
        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)

        # (b*w, c2, h)  (b*w,h,h) -> (b,w,c2,h) -> (b,c2,h,w)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)

        return self.gamma * (out_H + out_W) + x


class RCCAModule(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(RCCAModule, self).__init__()

        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.LayerNorm([inter_channels, 152, 272]),
                                   nn.LeakyReLU(0.3, inplace=False))

        self.cca = CC_module(in_dim=inter_channels)

        self.convb = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 3, padding=1, bias=False),
                                   nn.LayerNorm([out_channels, 152, 272]),
                                   nn.LeakyReLU(0.3, inplace=False))
        # self.bottleneck = nn.Sequential(
        #     nn.Conv2d(in_channels + in_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
        #     nn.LayerNorm([out_channels, 152, 272]),
        #     nn.LeakyReLU(0.3, inplace=True),
        #     nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        # )

    def forward(self, x, recurrence=2):

        output = self.conva(x)

        for i in range(recurrence):
            output = self.cca(output)

        output = self.convb(output)

        # output = self.bottleneck(torch.cat([x, output], 1))

        return output*x + x
