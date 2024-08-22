import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Parameter, Softmax
import torchvision
from tools.base_model import BaseModel
from tools.helpers import initialize_weights
from itertools import chain


def softplus_feature_map(x):
    return torch.nn.functional.softplus(x)


'''
-> BackBone Resnet
'''

class Resnet(nn.Module):
    def __init__(self, in_channels, backbone, out_channels_gcn=(85, 128), pretrained=True, kernel_sizes=(5, 7)):
        super(Resnet, self).__init__()
        resnet = getattr(torchvision.models, backbone)(pretrained)

        if in_channels == 3: conv1 = resnet.conv1
        else: conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial = nn.Sequential(conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        if not pretrained: initialize_weights(self)

    def forward(self, x):
        x = self.initial(x)
        conv1_sz = (x.size(2), x.size(3))
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4, conv1_sz


class PAM_Module(nn.Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super(PAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.softplus_feature = softplus_feature_map
        self.eps = eps

        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, height, width = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.softplus_feature(Q).permute(-3, -1, -2)
        K = self.softplus_feature(K)

        KV = torch.einsum("bmn, bcn->bmc", K, V)

        a = torch.sum(K, dim=-1)
        norm = 1 / torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps)

        weight_value = torch.einsum("bnm, bmc, bn->bcn", Q, KV, norm)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (x + self.gamma * weight_value).contiguous()


'''
-> Global Convolutionnal Network
'''

class GCN_Block(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels):
        super(GCN_Block, self).__init__()

        assert kernel_size % 2 == 1, 'Kernel size must be odd'
        self.conv11 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0))
        self.conv12 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, kernel_size//2))

        self.conv21 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, kernel_size//2))
        self.conv22 = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0))
        initialize_weights(self)

    def forward(self, x):
        x1 = self.conv11(x)
        x1 = self.conv12(x1)
        x2 = self.conv21(x)
        x2 = self.conv22(x2)

        x = x1 + x2
        return x

class BR_Block(nn.Module):
    def __init__(self, num_channels):
        super(BR_Block, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        initialize_weights(self)

    def forward(self, x):
        identity = x
        # x = self.conv1(self.relu1(self.bn1(x)))
        # x = self.conv2(self.relu2(self.bn2(x)))
        x = self.conv2(self.relu2(self.conv1(x)))
        x += identity
        return x

class GCN(BaseModel):
    def __init__(self, num_classes, in_channels=3, pretrained=True, use_resnet_gcn=False, backbone='resnet50', use_deconv=False,
                    num_filters=11, freeze_bn=False, **_):
        super(GCN, self).__init__()
        self.use_deconv = use_deconv
        if use_resnet_gcn:
            self.backbone = ResnetGCN(in_channels, backbone=backbone)
        else:
            self.backbone = Resnet(in_channels, pretrained=pretrained, backbone=backbone)

        if (backbone == 'resnet34' or backbone == 'resnet18'): resnet_channels = [64, 128, 256, 512]
        else: resnet_channels = [256, 512, 1024, 2048]
        
        self.gcn1 = GCN_Block(num_filters, resnet_channels[0], num_classes)
        self.br1 = BR_Block(num_classes)
        self.gcn2 = GCN_Block(num_filters, resnet_channels[1], num_classes)
        self.br2 = BR_Block(num_classes)
        self.gcn3 = GCN_Block(num_filters, resnet_channels[2], num_classes)
        self.br3 = BR_Block(num_classes)
        self.gcn4 = GCN_Block(num_filters, resnet_channels[3], num_classes)
        self.br4 = BR_Block(num_classes)
        self.attention4 = PAM_Module(resnet_channels[3])
        self.attention3 = PAM_Module(resnet_channels[2])
        self.attention2 = PAM_Module(resnet_channels[1])
        self.attention1 = PAM_Module(resnet_channels[0])

        self.br5 = BR_Block(num_classes)
        self.br6 = BR_Block(num_classes)
        self.br7 = BR_Block(num_classes)
        self.br8 = BR_Block(num_classes)
        self.br9 = BR_Block(num_classes)

        if self.use_deconv:
            self.decon1 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)
            self.decon2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)
            self.decon3 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)
            self.decon4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)
            self.decon5 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)
        self.final_conv = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        if freeze_bn: self.freeze_bn()
        # if freeze_backbone:
        #     set_trainable([self.backbone], False)

    def forward(self, x):
        x1, x2, x3, x4, conv1_sz = self.backbone(x)

        x1 = self.br1(self.gcn1(self.attention1(x1)))
        x2 = self.br2(self.gcn2(self.attention2(x2)))
        x3 = self.br3(self.gcn3(self.attention3(x3)))
        x4 = self.br4(self.gcn4(self.attention4(x4)))

        if self.use_deconv:
            # Padding because when using deconv, if the size is odd, we'll have an alignment error
            x4 = self.decon4(x4)
            if x4.size() != x3.size(): x4 = self._pad(x4, x3)
            x3 = self.decon3(self.br5(x3 + x4))
            if x3.size() != x2.size(): x3 = self._pad(x3, x2)
            x2 = self.decon2(self.br6(x2 + x3))
            x1 = self.decon1(self.br7(x1 + x2))

            x = self.br9(self.decon5(self.br8(x1)))
        else:
            x4 = F.interpolate(x4, size=x3.size()[2:], mode='bilinear', align_corners=True)
            x3 = F.interpolate(self.br5(x3 + x4), size=x2.size()[2:], mode='bilinear', align_corners=True)
            x2 = F.interpolate(self.br6(x2 + x3), size=x1.size()[2:], mode='bilinear', align_corners=True)
            x1 = F.interpolate(self.br7(x1 + x2), size=conv1_sz, mode='bilinear', align_corners=True)

            x = self.br9(F.interpolate(self.br8(x1), size=x.size()[2:], mode='bilinear', align_corners=True))
        return self.final_conv(x)

    def _pad(self, x_topad, x):
        pad = (x.size(3) - x_topad.size(3), 0, x.size(2) - x_topad.size(2), 0)
        x_topad = F.pad(x_topad, pad, "constant", 0)
        return x_topad

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return [p for n, p in self.named_parameters() if 'backbone' not in n]

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

