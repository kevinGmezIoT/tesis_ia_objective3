import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from torch.nn.parameter import Parameter

import torch.nn as nn

def gem(x, p=6.5, eps=1e-6):
    # print('x-',x.shape)
    # print('xpow-',x.clamp(min=eps).pow(p).shape)
    # print(F.avg_pool2d(x.clamp(min=eps).pow(p), (1, x.size(-1))).shape)
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (1, x.size(-1))).pow(1./p)

class GeM(nn.Module):

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.p.requires_grad = True # Allow learning GeM power
        self.eps = eps

    def forward(self, x):
        # print('p-',self.p)
        return gem(x, p=self.p, eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class Temporal(nn.Module):
    def __init__(self, inplanes, planes, bias=False, **kwargs):
        super(Temporal, self).__init__()

    def forward(self, x):
        
        out = torch.max(x, 2)[0]
        return out


class BasicConv3d_p(nn.Module):
    def __init__(self, inplanes, planes, kernel=3, bias=False, p=2, FM=False, **kwargs):
        super(BasicConv3d_p, self).__init__()
        self.p = p
        self.fm = FM
        self.convdl = nn.Conv3d(inplanes, planes, kernel_size=(kernel, kernel, kernel), bias=bias, padding=((kernel-1)//2, (kernel-1)//2, (kernel-1)//2))
        self.convdg = nn.Conv3d(inplanes, planes, kernel_size=(kernel, kernel, kernel), bias=bias, padding=((kernel-1)//2, (kernel-1)//2, (kernel-1)//2))
    def forward(self, x):
        n, c, t, h, w = x.size()
        scale = h//self.p
        # print('p-',x.shape,n, c, t, h, w,'scale-',scale)
        feature = list()
        for i in range(self.p):
            temp = self.convdl(x[:,:,:,i*scale:(i+1)*scale,:])
            # print(temp.shape,i*scale,(i+1)*scale)
            feature.append(temp)

        outl = torch.cat(feature, 3)
        # print('outl-',outl.shape)
        outl = F.leaky_relu(outl, inplace=True)

        outg = self.convdg(x)
        outg = F.leaky_relu(outg, inplace=True)
        # print('outg-',outg.shape)
        if not self.fm:
            # print('1-1')
            out = outg + outl
        else:
            # print('1-2')
            out = torch.cat((outg, outl), dim=3)
        return out


class BasicConv3d(nn.Module):
    def __init__(self, inplanes, planes, dilation=1, bias=False, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 3, 3), bias=bias, dilation=(dilation, 1, 1), padding=(dilation, 1, 1))

    def forward(self, x):
        out = self.conv1(x)
        out = F.leaky_relu(out, inplace=True)
        return out

class LocaltemporalAG(nn.Module):
    def __init__(self, inplanes, planes, dilation=1, bias=False, **kwargs):
        super(LocaltemporalAG, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), stride=(3,1,1), bias=bias,padding=(0, 0, 0))

    def forward(self, x):
        out1 = self.conv1(x)
        out = F.leaky_relu(out1, inplace=True)
        return out

class C3D_VGG(nn.Module):

    def __init__(self, hidden_dim=256, num_classes=74):
        super(C3D_VGG, self).__init__()
        self.hidden_dim = hidden_dim
        _set_channels = [32, 64, 128, 256]

        # --------------------------------  2d gei---------------------------------------
        self.conv2dlayer1a = BasicConv3d(1, _set_channels[0], kernel=3)
        # self.conv2dlayer1b = BasicConv3d(_set_channels[0], _set_channels[0])
        self.pool2d1 = LocaltemporalAG(_set_channels[0], _set_channels[0])


        self.conv2dlayer2a = BasicConv3d_p(_set_channels[0], _set_channels[1])
        # self.conv2dlayer2b = BasicConv3d(_set_channels[1], _set_channels[1])
        self.pool2d2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2dlayer25a_3d = BasicConv3d_p(_set_channels[1], _set_channels[2])
        self.conv2dlayer25b_3d = BasicConv3d_p(_set_channels[2], _set_channels[2])

        self.conv2dlayer3a_3d = BasicConv3d_p(_set_channels[2], _set_channels[3])
        self.conv2dlayer3b_3d = BasicConv3d_p(_set_channels[3], _set_channels[3], FM=True)

        self.fpb3d = Temporal(_set_channels[3], _set_channels[3])

        self.Gem = GeM()


        self.bin_numgl = [1, 2, 4, 8, 16] # Pyramid of bins for robustness
        self.fc_bin = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(sum(self.bin_numgl), _set_channels[3], self.hidden_dim)))
                    ])
        
        # BNNeck: BatchNorm before classification helps convergence
        self.bn = nn.BatchNorm1d(self.hidden_dim) 
        # Per-Bin Classification Head
        self.fc_id = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(sum(self.bin_numgl), self.hidden_dim, num_classes)))
        ])
                



        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        # print(x.shape)
        if x.dim() == 4:
            x = x.unsqueeze(1)
        n, c, t, h, w = x.size()
        if t == 1:
            x = x.repeat(1, 1, 3, 1, 1)
        elif t == 2:
            x = x.repeat(1, 1, 2, 1, 1)
        elif t == 3:
            x = torch.cat((x,x[:,:,0:1,:,:]),dim=2)
        # print(x.shape)

        # ----------------2d--------------------
        x2d = self.conv2dlayer1a(x)
        # x2d = self.conv2dlayer1b(x2d)
        x2d = self.pool2d1(x2d)
        # print('pool2d1-',x2d.shape)
        x2d = self.conv2dlayer2a(x2d)
        # x2d = self.conv2dlayer2b(x2d)
        x2d = self.pool2d2(x2d)
        # print('pool2d2-',x2d.shape)

        x2d = self.conv2dlayer25a_3d(x2d)
        x2d = self.conv2dlayer25b_3d(x2d)

        x2da3d = self.conv2dlayer3a_3d(x2d)
        # print('conv2dlayer3a_3d-',x2da3d.shape)
        x2db3d = self.conv2dlayer3b_3d(x2da3d)
        # print('conv2dlayer3b_3d-',x2db3d.shape)



        x2db3d = self.fpb3d(x2db3d)
        # print('x2db-',x2db3d.shape)

        # xgem = self.Gem(x2db3d)
        # print('xgem-',xgem.shape)

        _, c2d, _, _ = x2db3d.size()

        feature = list()
        for num_bin in self.bin_numgl:
            z = x2db3d.view(n, c2d, num_bin, -1).contiguous()
            # z1 = z.mean(3) + z.max(3)[0]
            # print('z1-',z1.shape)
            z2 = self.Gem(z).squeeze(-1)
            # print('z2-',z2.shape)
            feature.append(z2)
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()
        # print('feature',feature.shape)
        feature = feature.matmul(self.fc_bin[0])
        feature = feature.permute(1, 0, 2).contiguous()
        
        # Per-Bin Classification (SOTA GaitGL style)
        # feature: [batch, bins, dim]
        # self.fc_id[0]: [bins, dim, num_classes]
        n_b, n_bins, n_dim = feature.size()
        feat_flat = feature.view(-1, n_dim)
        feat_bn = self.bn(feat_flat)
        feat_bn = feat_bn.view(n_b, n_bins, n_dim)
        
        # Optimize memory by using einsum instead of broadcasted matmul
        # [batch, bins, dim] * [bins, dim, num_classes] -> [batch, bins, num_classes]
        logits = torch.einsum('nbd,bdc->nbc', feat_bn, self.fc_id[0])

        # L2 Normalization RESTORED for Triplet branch stability
        feature = F.normalize(feature, p=2, dim=-1)

        return feature, logits


def params_count(net):
    list1 = []
    for p in net.parameters():
        # print('p-',p.shape)
        list1.append(p)
    # print('len(net.parameters)',len(list1))
    n_parameters = sum(p.numel() for p in net.parameters())
    print('-----Model param: {:.5f}M'.format(n_parameters / 1e6))
    # print('-----Model memory: {:.5f}M'.format(n_parameters/1e6))
    return n_parameters


def c3d_vgg_Fusion(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = C3D_VGG(**kwargs)
    return model


if __name__ == "__main__":
    net = c3d_vgg_Fusion(hidden_dim=256, num_classes=74)
    print(params_count(net))
    with torch.no_grad():
        # x = torch.ones(4*3*16*64*44).reshape(4,3,16,64,44)
        x = torch.ones(4 * 1 * 32 * 64 * 44).reshape(4, 1, 32, 64, 44)
        # a = Variable(a.cuda)
        print('x=', x.shape)
        # a,b = net(x)
        # print('a,b=',a.shape,b.shape)
        a,_ = net(x)
        print('a,b=', a.shape)