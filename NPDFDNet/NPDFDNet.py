

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def idwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return idwt_init(x)

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class Convgroup(nn.Module):
    def __init__(self,f):
        super(Convgroup, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.relu(conv1))
        conv3 = self.conv3(self.relu(conv2))
        return conv3

class MESA(nn.Module):
    def __init__(self, n_feats):
        super(MESA, self).__init__()
        f = n_feats // 4
        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_s = nn.Conv2d(f, f, kernel_size=3,stride=2, padding=0)

        self.Convgroup1 = Convgroup(f)
        self.Convgroup2 = Convgroup(f)
        self.Convgroup3 = Convgroup(f)

        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.conv_1 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1 = (self.conv1(x))
        c_s = self.conv_s(c1)

        maxpool1 = F.max_pool2d(c_s, kernel_size=7, stride=3)
        maxpool2 = F.max_pool2d(c_s, kernel_size=7, stride=5)
        maxpool3 = F.max_pool2d(c_s, kernel_size=7, stride=7)

        cg1 = self.Convgroup1(maxpool1)
        cg2 = self.Convgroup2(maxpool2)
        cg3 = self.Convgroup3(maxpool3)

        cg1 = F.interpolate(cg1, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cg2 = F.interpolate(cg2, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cg3 = F.interpolate(cg3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        out_fus = c1+cg1+cg2+cg3
        out_fus = self.conv_1(out_fus)

        m = self.sigmoid(out_fus)

        return x * m

class NPSamlping(nn.Module):
    def __init__(self, base_filter):
        super(NPSamlping, self).__init__()

        self.softmax = nn.Softmax(dim=-1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2, bias=False)

        self.sampling = nn.Conv2d(in_channels=1, out_channels=base_filter, kernel_size=32, stride=32, padding=0, bias=False)



    def forward(self, x):

        b, c, h, w = x.shape
        V = self.conv1(x)
        Q = self.conv3(x)
        K = self.conv5(x)

        V = V.view(b, c, -1)
        Q = Q.view(b, c, -1)
        K = K.view(b, c, -1)

        Q = torch.nn.functional.normalize(Q, dim=-1)
        K = torch.nn.functional.normalize(K, dim=-1)

        att = torch.matmul(Q, K.permute(0, 2, 1))

        att = self.softmax(att)
        out = torch.matmul(att, V).view(b, c, h, w)
        output = out + x

        mesurment = self.sampling(output)
        return mesurment

class DFDB_Block(nn.Module):
    def __init__(self):
        super(DFDB_Block, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=80, out_channels=64, kernel_size=3, stride=1, padding=2,dilation=2, bias=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=112, out_channels=64, kernel_size=3, stride=1, padding=3,dilation=3, bias=True))
        #self.conv4 = nn.Sequential(nn.Conv2d(in_channels=160, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True))
        self.sq1 = nn.Conv2d(in_channels=64, out_channels=48, kernel_size=1, stride=1, padding=0, bias=False)
        self.sq2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        self.sq3 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False)
        self.msea = MESA(64)
        self.confusion = nn.Conv2d(in_channels=160, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity_data = x
        identity_data_16 = self.sq3(identity_data)
        output1 = self.conv1(x)
        #output1 = self.bn1(output1)
        output1 = self.relu(output1)
        output1_1 = torch.cat([identity_data_16, output1], 1)


        output2 = self.conv2(output1_1)
        #output2 = self.bn2(output2)
        output2 = self.relu(output2)
        output2 = self.msea(output2)
        identity_data_32 = self.sq2(identity_data)
        output1_16 = self.sq3(output1)
        output2_2 = torch.cat([identity_data_32, output1_16, output2], 1)

        output3 = self.conv3(output2_2)
        #output3 = self.bn3(output3)
        output3 = self.relu(output3)
        identity_data_48 = self.sq1(identity_data)
        output1_32 = self.sq2(output1)
        output2_16 = self.sq3(output2)
        output3_3 = torch.cat([identity_data_48, output1_32, output2_16, output3], 1)

        #output4 = self.conv3(output3_3)
        output = self.confusion(output3_3)
        output = self.relu(output)
        output = torch.add(output, identity_data)
        return output

class NPDFDNet(nn.Module):
    def __init__(self, base_filter):
        super(NPDFDNet, self).__init__()

        self.NPSamlping = NPSamlping(base_filter)

        self.initialization1 = nn.Conv2d(in_channels=base_filter, out_channels=1024, kernel_size=1, stride=1, padding=0,bias=False)

        self.initialization2 = nn.PixelShuffle(32)


        self.DWT1 = DWT();
        self.IWT1 = IDWT();


        self.ex1 =  nn.Conv2d(in_channels = 4,out_channels= 64,kernel_size=1, stride=1, padding=0, bias=True)
        self.sq1 =  nn.Conv2d(in_channels = 64,out_channels= 4,kernel_size=1, stride=1, padding=0, bias=True)


        #self.getFactor = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.residual1 = self.make_layer(DFDB_Block)
        self.residual2 = self.make_layer(DFDB_Block)
        self.residual3 = self.make_layer(DFDB_Block)
        self.residual4 = self.make_layer(DFDB_Block)
        self.residual5 = self.make_layer(DFDB_Block)
        self.residual6 = self.make_layer(DFDB_Block)
        self.residual7 = self.make_layer(DFDB_Block)
        #self.residual8 = self.make_layer(MRB_Block)

        self.cat = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True)

        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.NPSamlping(x)

        out = self.initialization1(out)
        LR = self.initialization2(out)

        dwt1 = self.DWT1(LR)

        dwt1 = self.ex1(dwt1)

        out1 = self.residual1(dwt1)
        out2 = self.residual2(out1)
        out3 = self.residual3(out2)
        out4 = self.residual4(out3)
        out5 = self.residual5(out4)
        out6 = self.residual6(out5)
        out7 = self.residual7(out6)
        #out8 = self.residual8(out7)

        #out = torch.cat([dwt1, out1, out2, out3, out4, out5, out6, out7, out8], 1)
        out = torch.cat([dwt1, out1, out2, out3, out4, out5, out6,out7], 1)
        outcat =self.relu(self.cat(out))

        outadd = torch.add(dwt1, outcat)

        out3_1 = self.conv3_1(outadd)

        outsq1 = self.sq1(out3_1)

        iwt1 = self.IWT1(outsq1)

        #out = self.relu(self.add(out))
        output = self.conv3_2(iwt1)
        #out = self.ca(out) * out

       # out = self.output(output)

        return output

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')