import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbones.resnet_for_bes import FCNHead, UP_Conv, resnet18

# feature add

def softplus_feature_map(x):
    return torch.nn.functional.softplus(x)


# 返回一个tuple 含x 和与boundary GT做loss的bl 先修改了只输出一个 需要的话要改回
class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
        logging.info("Global Average Pooling Initialized")

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)


class ConvBnReLU(nn.Sequential):
    def __init__(
            self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(out_ch))

        if relu:
            self.add_module("relu", nn.ReLU())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def Upsample(x, size):
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True))
    return block


# ASPP将全局context纳入模型，最后一个feature map用GAP
class AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AsppPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)
        return Upsample(pool, (h, w))  # GAP后的分辨率1×1，要上采样才能concat


class ASPP_Module(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP_Module, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = tuple(atrous_rates)  # tuple元组
        self.b0 = nn.Sequential(    # 第一个feature map 1×1
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)  # 3×3
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = AsppPooling(in_channels, out_channels)  # 最后一个feature map GAP

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return y


class BE_Module(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch5, mid_ch, out_ch, n_class):
        super(BE_Module, self).__init__()

        self.convb_1 = ConvBnReLU(in_ch1, mid_ch, kernel_size=1, stride=1, padding=0, dilation=1)  # 用来统一channel
        self.convb_2 = ConvBnReLU(in_ch2, mid_ch, kernel_size=1, stride=1, padding=0, dilation=1)
        self.convb_5 = ConvBnReLU(in_ch5, mid_ch, kernel_size=1, stride=1, padding=0, dilation=1)
        self.convbloss = nn.Conv2d(mid_ch, n_class, kernel_size=1, bias=False)
        boundary_ch = 3 * mid_ch
        self.boundaryconv = ConvBnReLU(boundary_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)

    def forward(self, l1, l2, l5):
        # 统一channel,F5需要上采
        l1_b = self.convb_1(l1)
        l2_b = self.convb_2(l2)
        l5_b = self.convb_5(l5)
        l5_b = F.interpolate(l5_b, l1.size()[2:], mode='bilinear', align_corners=True)

        l1_bl = self.convbloss(l1_b)  # l1的boundary
        l2_bl = self.convbloss(l2_b)

        l5_bl = self.convbloss(l5_b)

        b = torch.cat((l1_b, l2_b, l5_b), dim=1)  # 输出的boundary feature Fb
        b = self.boundaryconv(b)

        c_boundaryloss = l1_bl + l2_bl + l5_bl

        return b, c_boundaryloss


class MSF_Module(nn.Module):
    def __init__(self, in_ch, mid_ch1, cat_ch, mid_ch2, out_ch):
        super(MSF_Module, self).__init__()

        self.input1 = ConvBnReLU(in_ch[0], mid_ch1, kernel_size=1, stride=1, padding=0, dilation=1)
        self.input2 = ConvBnReLU(in_ch[1], mid_ch1, kernel_size=1, stride=1, padding=0, dilation=1)
        self.input3 = ConvBnReLU(in_ch[2], mid_ch1, kernel_size=1, stride=1, padding=0, dilation=1)

        self.fusion1 = nn.Sequential(
            ConvBnReLU(cat_ch, mid_ch2, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.Conv2d(mid_ch2, mid_ch2, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Sigmoid(),
            GlobalAvgPool2d()
        )

        self.fusion2 = nn.Sequential(
            ConvBnReLU(cat_ch, mid_ch2, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.Conv2d(mid_ch2, out_ch, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Sigmoid(),
            GlobalAvgPool2d()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, l3, l4, l5):
        f3 = self.input1(l3)
        f4 = self.input2(l4)
        f5 = self.input3(l5)

        w1 = torch.cat((f4, f5), dim=1)
        w1 = self.fusion1(w1).unsqueeze(2).unsqueeze(3).expand_as(f5)
        m1 = (1 - w1) * f4 + w1 * f5

        w2 = torch.cat((m1, f3), dim=1)
        w2 = self.fusion2(w2).unsqueeze(2).unsqueeze(3).expand_as(f5)
        m2 = (1 - w2) * f3 + w2 * m1

        return m2


class BES_Module(nn.Module):
    def __init__(self, f5_in, mul_ch):
        super(BES_Module, self).__init__()
        aspp_out = 5 * f5_in // 8
        self.aspp = ASPP_Module(f5_in, atrous_rates=[12, 24, 36])
        self.f5_out = ConvBnReLU(aspp_out, mul_ch, kernel_size=3, stride=1, padding=1, dilation=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, f5, fb, ff):
        aspp = self.aspp(f5)
        f5 = self.f5_out(aspp)
        f5 = F.interpolate(f5, fb.size()[2:], mode='bilinear', align_corners=True)
        f5_guide = torch.mul(f5, fb)  # 对应位置相乘
        ff_guide = torch.mul(ff, fb)
        fe = ff + ff_guide + f5_guide

        return fe


class PAM_Module(nn.Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super(PAM_Module, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.in_places = in_places
        self.softplus_feature = softplus_feature_map
        self.eps = eps

        self.query_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, height, width = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.softplus_feature(Q).permute(-3, -1, -2)
        K = self.softplus_feature(K)

        KV = torch.einsum("bmn, bcn->bmc", K, V)

        norm = 1 / torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps)

        weight_value = torch.einsum("bnm, bmc, bn->bcn", Q, KV, norm)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (x + self.gamma * weight_value).contiguous()


class BESNet(nn.Module):
    def __init__(self, nclass, backbone='resnet18', aux=True, norm_layer=nn.BatchNorm2d, pretrained=True):
        super(BESNet, self).__init__()

        self.aux = aux  # 辅助loss
        resnet = eval(backbone)(pretrained=pretrained)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.attention = PAM_Module(64)

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.BoundaryExtraction = BE_Module(64, 64, 512, 64, 128, 1)  # in_ch1, in_ch2, in_ch5, mid_ch, out_ch, n_class
        self.Fusion = MSF_Module([128, 256, 512], 128, 256, 128, 128)  # in_ch(元组), mid_ch1, cat_ch, mid_ch2, out_ch
        self.up = UP_Conv(128, 128)  # ch_in, ch_out
        self.Enhance = BES_Module(512, 128)  # f5_in, mul_ch
        self.head = FCNHead(192, nclass, norm_layer)  # in_channels, out_channels, norm_layer=nn.BatchNorm2d
        if self.aux:
            self.auxlayer = FCNHead(256, nclass, norm_layer)

    def forward(self, x):

        imsize = x.size()[2:]

        c0 = x = self.layer0(x)
        c1 = x = self.layer1(x)
        c2 = x = self.layer2(x)
        c3 = x = self.layer3(x)
        c4 = x = self.layer4(x)

        b, c_boundaryloss = self.BoundaryExtraction(c0, c1, c4)

        f = self.Fusion(c2, c3, c4)
        f = self.up(f)

        x = self.Enhance(c4, b, f)
        # x = x + self.attention(c1)
        x = torch.cat((x, self.attention(c1)), dim=1)

        x = self.head(x)
        x = Upsample(x, imsize)

        return x

        # outputs = [x]
        # if self.aux:
        #     auxout = self.auxlayer(c3)
        #     auxout = Upsample(auxout, imsize)
        #     outputs.append(auxout)
        #
        # if self.training and self.aux:
        #     outputs.append(c_boundaryloss)
        #     return tuple(outputs)
        # return x, c_boundaryloss