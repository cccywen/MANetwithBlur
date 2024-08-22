import torch.nn.functional as F
from torch.nn import Module, Conv2d, Parameter, Softmax
import torch
from torchvision import models
from torch import nn

from functools import partial


nonlinearity = partial(F.relu, inplace=True)


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


def softplus_feature_map(x):
    return torch.nn.functional.softplus(x)


def conv3otherRelu(in_planes, out_planes, kernel_size=None, stride=None, padding=None):
    # 3x3 convolution with padding and relu
    if kernel_size is None:
        kernel_size = 3
    assert isinstance(kernel_size, (int, tuple)), 'kernel_size is not in (int, tuple)!'

    if stride is None:
        stride = 1
    assert isinstance(stride, (int, tuple)), 'stride is not in (int, tuple)!'

    if padding is None:
        padding = 1
    assert isinstance(padding, (int, tuple)), 'padding is not in (int, tuple)!'

    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.ReLU(inplace=True)  # inplace=True
    )


class PAM_Module(Module):
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

        Q_H = Q.permute(0, 3, 1, 2).contiguous().view(batch_size * width, -1, height).permute(0, 2, 1)
        Q_W = Q.permute(0, 2, 1, 3).contiguous().view(batch_size * height, -1, width).permute(0, 2, 1)
        K_H = K.permute(0, 3, 1, 2).contiguous().view(batch_size * width, -1, height)
        K_W = K.permute(0, 2, 1, 3).contiguous().view(batch_size * height, -1, width)
        V_H = V.permute(0, 3, 1, 2).contiguous().view(batch_size * width, -1, height)
        V_W = V.permute(0, 2, 1, 3).contiguous().view(batch_size * height, -1, width)

        # Q = self.softplus_feature(Q).permute(-3, -1, -2)
        # K = self.softplus_feature(K)

        # KV = torch.einsum("bmn, bcn->bmc", K, V)
        KV_H = torch.einsum("bmn, bcn->bmc", K_H, V_H)
        KV_W = torch.einsum("bmn, bcn->bmc", K_W, V_W)

        norm_H = 1 / torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps)

        weight_value = torch.einsum("bnm, bmc, bn->bcn", Q, KV, norm)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (x + self.gamma * weight_value).contiguous()


class CCAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CCAttention, self).__init__()
        # Q K 1×1卷积降维生成
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # V 1×1卷积相当于恒等映射
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # 沿channel softmax
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.softplus_feature = softplus_feature_map
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        Q = self.query_conv(x)
        # contiguous()先copy一份张量在内存中的地址，然后将地址按照形状改变后的张量的语义进行排列,在不改变原tensor的情况下改变..
        Q_H = self.softplus_feature(Q.permute(0,3,1,2).contiguous().view(m_batchsize*width, -1, height).permute(0, 2, 1))
        Q_W = self.softplus_feature(Q.permute(0,2,1,3).contiguous().view(m_batchsize*height, -1, width).permute(0, 2, 1))
        K = self.key_conv(x)

        K_H = self.softplus_feature(K.permute(0,3,1,2).contiguous().view(m_batchsize*width, -1, height))
        K_W = self.softplus_feature(K.permute(0,2,1,3).contiguous().view(m_batchsize*height, -1, width))
        V = self.value_conv(x)
        V_H = V.permute(0,3,1,2).contiguous().view(m_batchsize*width, -1, height)
        V_W = V.permute(0,2,1,3).contiguous().view(m_batchsize*height, -1, width)
        energy_H = (torch.bmm(Q_H, K_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(Q_W, K_W).view(m_batchsize, height, width, width)
        concate = torch.cat([energy_H, energy_W], 3)

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width, height, height)

        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(V_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(V_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x


class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),nn.ReLU(inplace=False))
        self.cca = CCAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),nn.ReLU(inplace=False))

        self.bottle = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=False),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x, recurrence=1):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottle(torch.cat([x, output], 1))

        return output


class CAM_Module(Module):
    def __init__(self):
        super(CAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        batch_size, chnnels, height, width = x.shape
        proj_query = x.view(batch_size, chnnels, -1)
        proj_key = x.view(batch_size, chnnels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(batch_size, chnnels, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, chnnels, height, width)

        out = self.gamma * out + x
        return out


class PAM_CAM_Layer(nn.Module):
    def __init__(self, in_ch):
        super(PAM_CAM_Layer, self).__init__()
        self.PAM = PAM_Module(in_ch)
        self.CAM = CAM_Module()

    def forward(self, x):
        return self.PAM(x) + self.CAM(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class MANet(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):  # 改backbone
        super(MANet, self).__init__()
        self.name = 'MANet'

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=pretrained)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.attention4 = RCCAModule(filters[3], 2048)
        self.attention3 = RCCAModule(filters[2], 1024)
        self.attention2 = RCCAModule(filters[1], 512)
        self.attention1 = RCCAModule(filters[0], 256)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        # e1, e2, e3, e4 = self.backbone(x)
        x1 = self.firstconv(x)
        x1 = self.firstbn(x1)
        x1 = self.firstrelu(x1)
        x1 = self.firstmaxpool(x1)
        e1 = self.encoder1(x1)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e4 = self.attention4(e4)
        # a3 = self.attention3(e3)
        # a2 = self.attention3(e2)
        # a1 = self.attention3(e1)

        # Decoder
        d4 = self.decoder4(e4) + self.attention3(e3)
        d3 = self.decoder3(d4) + self.attention2(e2)
        d2 = self.decoder2(d3) + self.attention1(e1)
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out


def manet_base(pretrained=True, num_classes=4, weight_path='/home/caoyiwen/pretrained weights/resnet50-timm.pth'):
    model = MANet(num_classes=num_classes)
    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path)
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model
