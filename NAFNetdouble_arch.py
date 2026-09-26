# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''
import sys
sys.path.append("D:/lkj/Intrinsic/NAFNet-main")
from re import T
from turtle import forward
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base
from basicsr.models.archs.pac_arch import *

import numpy
import csv
import cv2
import matplotlib.pyplot as plt


class SimpleGate(nn.Module):
    def forward(self, x):
        # 将x分为2块
        x1, x2 = x.chunk(2, dim=1)
        # Tempx1 = x1.cpu().detach().numpy()
        # Tempx2 = x2.cpu().detach().numpy()
        # for i in range(Tempx1.shape[0]):
        #     sx1 = Tempx1[i].mean(axis=0)
        #     sx2 = Tempx2[i].mean(axis=0)
        #     plt.imshow(sx1,cmap='gray')
        #     plt.show()
        #     plt.imshow(sx2,cmap='gray')
        #     plt.show()

        return x1 * x2


class SpatialAttention(nn.Module):
    def __init__(self, in_channel=2, out_channel=1, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        # output = self.sigmoid(output)
        return output


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand

        self.conv0 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True)
        self.conv1 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                               groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)

        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        # self.conv3 = nn.Conv2d(in_channels=dw_channel , out_channels=c, kernel_size=1, padding=0, stride=1,
        #                        groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # self.sca = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=True),
        # )
        # SimpleGate
        self.sg = SimpleGate()
        self.sp_t = SpatialAttention(kernel_size=5)

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        # self.conv4 = nn.Conv2d(in_channels=c, out_channels=c//2, kernel_size=1, padding=0, stride=1, groups=1,
        #                       bias=True)

        self.conv4_5 = nn.Conv2d(in_channels=ffn_channel, out_channels=ffn_channel // 2, kernel_size=1, padding=0,
                                 stride=1, groups=1,
                                 bias=False)

        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        # self.conv5 = nn.Conv2d(in_channels=c//2 , out_channels=c, kernel_size=1, padding=0, stride=1,
        #                       groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.sig = nn.Sigmoid()

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        x = self.conv0(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)

        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))

        # x = self.norm2(y)
        sp = self.sp_t(x)
        x = self.conv4_5(x) * sp

        x = self.conv5(x)
        #
        x = self.dropout2(x)
        return y + x * self.gamma
        # return y + x * self.gamma

        # x = self.conv4(self.norm2(y))
        # x = self.sg(x)
        # x = self.conv5(x)

        # x = self.dropout2(x)

        # return y + x * self.gamma


class NAFBlock2(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)

        self.conv2_3 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel // 2, kernel_size=1, padding=0,
                                 stride=1,
                                 groups=1,
                                 bias=True)

        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        # self.conv3 = nn.Conv2d(in_channels=dw_channel , out_channels=c, kernel_size=1, padding=0, stride=1,
        #                        groups=1, bias=True)

        self.sp_t = SpatialAttention(kernel_size=3)
        self.sp_t2 = SpatialAttention(kernel_size=1)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # self.sca = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=True),
        # )
        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)

        self.conv4_5 = nn.Conv2d(in_channels=ffn_channel, out_channels=ffn_channel // 2, kernel_size=1, padding=0,
                                 stride=1, groups=1,
                                 bias=False)

        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        # self.conv5 = nn.Conv2d(in_channels=ffn_channel , out_channels=c, kernel_size=1, padding=0, stride=1,
        #                        groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)

        sp_1 = self.sp_t(x)
        x = self.conv2_3(x) * sp_1
        # x = self.sg(x)

        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        # x = self.sg(x)

        sp2 = self.sp_t2(x)
        x = self.conv4_5(x) * sp2

        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFBlock3(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)

        self.conv2_3 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel // 2, kernel_size=1, padding=0,
                                 stride=1,
                                 groups=1,
                                 bias=True)

        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        # self.conv3 = nn.Conv2d(in_channels=dw_channel , out_channels=c, kernel_size=1, padding=0, stride=1,
        #                        groups=1, bias=True)

        self.sp_t = SpatialAttention(kernel_size=5)
        self.sp_t2 = SpatialAttention(kernel_size=7)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # self.sca = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=True),
        # )
        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)

        self.conv4_5 = nn.Conv2d(in_channels=ffn_channel, out_channels=ffn_channel // 2, kernel_size=1, padding=0,
                                 stride=1, groups=1,
                                 bias=False)

        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        # self.conv5 = nn.Conv2d(in_channels=ffn_channel , out_channels=c, kernel_size=1, padding=0, stride=1,
        #                        groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)

        sp_1 = self.sp_t(x)
        sp_2 = self.sp_t2(x)
        xtemp = self.conv2_3(x)
        x = xtemp * sp_1 + xtemp * sp_2
        # x = self.sg(x)

        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        return y

        # x = self.conv4(self.norm2(y))
        # # x = self.sg(x)
        #
        # sp2 = self.sp_t2(x)
        # x = self.conv4_5(x) * sp2
        #
        # x = self.conv5(x)
        #
        # x = self.dropout2(x)

        # return y + x * self.gamma


class NAFNet_Double(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums1=[], dec_blk_nums2=[],
                 pac_R_size=None, pac_S_size=None):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending1 = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                 groups=1,
                                 bias=True)

        self.ending2 = nn.Conv2d(in_channels=width, out_channels=1, kernel_size=3, padding=1, stride=1, groups=1,
                                 bias=True)

        self.encoders = nn.ModuleList()
        self.decoders1 = nn.ModuleList()
        self.decoders2 = nn.ModuleList()
        self.up1s = nn.ModuleList()
        self.up2s = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.downs_help = nn.ModuleList()
        self.up_help = nn.ModuleList()

        self.usepacR = False
        self.usepacS = False

        chan = width

        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            ###
            self.downs_help.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )

            chan = chan * 2
        chan1 = chan2 = chan

        self.ext_temp1 = nn.Conv2d(chan1, chan1, 1, bias=False)
        self.ext_temp2 = nn.Conv2d(chan2, chan2, 1, bias=False)

        for num in dec_blk_nums1:
            self.up1s.append(
                nn.Sequential(
                    nn.Conv2d(chan1, chan1 * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )

            self.up_help.append(
                nn.Sequential(
                    nn.Conv2d(chan1, chan1 * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan1 = chan1 // 2
            self.decoders1.append(
                nn.Sequential(
                    # nn.Conv2d(chan1 * 2, chan1, 1, bias=False),
                    *[NAFBlock2(chan1) for _ in range(num)]
                )
            )

        for num in dec_blk_nums2:
            self.up2s.append(
                nn.Sequential(
                    nn.Conv2d(chan2, chan2 * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan2 = chan2 // 2
            self.decoders2.append(
                nn.Sequential(
                    *[NAFBlock3(chan2) for _ in range(num)]
                )
            )

        if pac_R_size != None:
            print('pacR:', pac_R_size)
            self.usepacR = True
            self.pac1 = PacConv2d(in_channels=img_channel, out_channels=img_channel, kernel_size=int(pac_R_size),
                                  stride=1, dilation=1, padding=int(pac_R_size) // 2,
                                  normalize_kernel=True, bias=False, native_impl=True)

        if pac_S_size != None:
            print('pacS:', pac_S_size)
            self.usepacS = True
            self.pac2 = PacConv2d(in_channels=1, out_channels=1, kernel_size=int(pac_S_size),
                                  stride=1, dilation=1, padding=int(pac_S_size) // 2,
                                  normalize_kernel=True, bias=False, native_impl=True)

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        if self.usepacS:
            meaninp = torch.mean(inp, dim=1, keepdim=True)
            # V,_ = torch.max(inp,dim=1,keepdim=True)

        encs = []
        encs2 = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        #
        # enct = self.downs_help[2](self.downs_help[1](self.downs_help[0](encs[0])))
        # encs2.append(enct)
        # enct = self.downs_help[1](encs[1])
        # encs2.append(enct)
        # enct = self.up_help[2](encs[2])
        # encs2.append(enct)
        # enct = self.up_help[3](self.up_help[2](self.up_help[1](encs[3])))
        # encs2.append(enct)

        # enct = self.downs_help[1](self.downs_help[0](encs[0]))
        # encs2.append(enct)
        # enct = encs[1]
        # encs2.append(enct)
        # enct = self.up_help[2](self.up_help[1](encs[2]))
        # encs2.append(enct)

        # original
        x1 = x2 = x
        x1 = self.ext_temp1(x)
        x2 = self.ext_temp2(x)
        # backbone

        for decoder, up, enc_skip in zip(self.decoders1, self.up1s, encs[::-1]):
            x1 = up(x1)
            x1 = x1 + enc_skip
            # x1 = torch.cat([x1,enc_skip],dim = 1)
            x1 = decoder(x1)
        # for decoder, up, enc_skip in zip(self.decoders1, self.up1s, encs[::-1]):
        #     x1 = up(x1)
        #     x1 = x1 + enc_skip
        #     x1 = decoder(x1)
        for decoder, up, enc_skip in zip(self.decoders2, self.up2s, encs[::-1]):
            x2 = up(x2)
            # x2 = x2 + enc_skip
            x2 = decoder(x2)

        x1 = self.ending1(x1)
        x1 = x1 + inp
        x2 = self.ending2(x2)

        # x2 = x2 + inp
        # bx1 = x1
        # bx2 = x2
        if self.usepacR:
            x1 = self.pac1(x1, x2)
        if self.usepacS:
            x2 = self.pac2(x2, meaninp)
        return [x1[:, :, :H, :W], x2[:, :, :H, :W]]
        # return [x1[:, :, :H, :W], x2[:, :, :H, :W],bx1[:, :, :H, :W],bx2[:, :, :H, :W]]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


# guidance block
class Guidace_Layer(nn.Module):
    def __init__(self, r, eps):
        super(Guidace_Layer, self).__init__()
        # kenerl
        kenerl = 1 / ((2 * r + 1) * (2 * r + 1)) * torch.ones(3, 1, 2 * r + 1, 2 * r + 1)
        kenerl = torch.reshape(kenerl, (3, 1, 2 * r + 1, 2 * r + 1))
        self.boxfilter = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=r, stride=1, padding=r,
                                   padding_mode='reflect', groups=3, bias=False)
        self.boxfilter.weight = nn.Parameter(kenerl, requires_grad=False)
        self.eps = eps

    def forward(self, I, P):
        t_meanI = self.boxfilter(I)
        t_meanP = self.boxfilter(P)
        t_corrI = self.boxfilter(I * I)
        t_corrIP = self.boxfilter(I * P)
        # 计算相关系数，计算IP的协方差cov和I的方差var
        varI = t_corrI - t_meanI * t_meanI
        covIP = t_corrIP - t_meanI * t_meanP
        # 计算系数参数a，b
        a = covIP / (varI + self.eps)
        b = t_meanP - a * t_meanI
        # 计算系数a，b的均值
        meanA = self.boxfilter(a)
        meanB = self.boxfilter(b)
        # 生输出矩阵
        q = meanA * I + meanB
        return q


# if __name__ == '__main__':
#
#
#     torch.cuda.set_device(0)
#     model = NAFNet_Double(img_channel=3, width=32, middle_blk_num=0, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums1=[2, 2, 2, 2], dec_blk_nums2=[2, 2, 2, 2],
#                  pac_R_size=3, pac_S_size=3).cuda()
#     from thop import profile
#
#     input = torch.randn(1, 3, 1024, 436).cuda()
#     flops, params = profile(model, inputs=(input,))
#
#     print("flops=" + str(flops))
#     print("params=" + str(params))
#
#     import resource
#     def using(point=""):
#         # print(f'using .. {point}')
#         usage = resource.getrusage(resource.RUSAGE_SELF)
#         global Total, LastMem
#
#         # if usage[2]/1024.0 - LastMem > 0.01:
#         # print(point, usage[2]/1024.0)
#         print(point, usage[2] / 1024.0)
#
#         LastMem = usage[2] / 1024.0
#         return usage[2] / 1024.0
#
#
#     img_channel = 3
#     width = 32
#
#     enc_blks = [2, 2, 2, 20]
#     middle_blk_num = 2
#     dec_blks = [2, 2, 2, 2]
#
#     print('enc blks', enc_blks, 'middle blk num', middle_blk_num, 'dec blks', dec_blks, 'width', width)
#
#     using('start . ')
#     net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
#                  enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
#
#     using('network .. ')
#
#     # for n, p in net.named_parameters()
#     #     print(n, p.shape)
#
#     inp = torch.randn((4, 3, 256, 256))
#
#     out = net(inp)
#     final_mem = using('end .. ')
#     # out.sum().backward()
#
#     # out.sum().backward()
#
#     # using('backward .. ')
#
#     # exit(0)
#
#     inp_shape = (3, 512, 512)
#
#     from ptflops import get_model_complexity_info
#
#     macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)
#
#     params = float(params[:-3])
#     macs = float(macs[:-4])
#
#     print(macs, params)
#
#     print('total .. ', params * 8 + final_mem)

