import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from models.initialization import init_weights
from models.BCRNN import Conv2dFT


#H_out = ⌊ (H_in+2×padding[0]−dilation[0]×(kernel_size[0]−1)−1)/stride[0] + 1⌋ 


def ConvBlock(input_dim, output_dim, kernel_size=3, stride=1, padding=1, dilation_rate=1, use_bn=1, convFT=0, slim=False):
    return [
        nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding, dilation=dilation_rate),
        nn.BatchNorm2d(output_dim),
        nn.ReLU(inplace=True)
    ]

class DilatedConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, use_bn=1, convFT=0, slim=False):
        super(DilatedConvBlock, self).__init__()
        self.layers_1 = nn.Sequential(*ConvBlock(input_dim, output_dim//2, kernel_size, stride, padding=1, dilation_rate=1))
        self.layers_2 = nn.Sequential(*ConvBlock(output_dim//2, output_dim//4, kernel_size, stride, padding=3, dilation_rate=3))
        self.layers_3 = nn.Sequential(*ConvBlock(output_dim//4, output_dim//8, kernel_size, stride, padding=6, dilation_rate=6))
        self.layers_4 = nn.Sequential(*ConvBlock(output_dim//8, output_dim//16, kernel_size, stride, padding=9, dilation_rate=9))
        self.layers_5 = nn.Sequential(*ConvBlock(output_dim//16, output_dim//16, kernel_size, stride, padding=12, dilation_rate=12))
        self.layers_1.apply(init_weights)
        self.layers_2.apply(init_weights)
        self.layers_3.apply(init_weights)
        self.layers_4.apply(init_weights)
        self.layers_5.apply(init_weights)
    
    def forward(self, inputs):
        x1 = self.layers_1(inputs)
        x2 = self.layers_2(x1)
        x3 = self.layers_3(x2)
        x4 = self.layers_4(x3)
        x5 = self.layers_5(x4)
        outputs = torch.cat([x1, x2, x3, x4, x5], 1)
        return outputs


class DownConvBlock(nn.Module):

    def __init__(
        self, 
        input_dim, 
        output_dim, 
        kernel_size=3,
        stride=1,
        padding=1, 
        use_bn=1, 
        pool=True,
        slim=False,
        convFT=0
    ):

        super(DownConvBlock, self).__init__()

        self.pool = pool
        self.pool_layer = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            # self.layers.append(nn.Conv2d(input_dim, input_dim, kernel_size=2, stride=2, padding=0))
        self.convBlock = DilatedConvBlock(input_dim, output_dim, kernel_size, stride, padding, use_bn, slim, convFT)

    def forward(self, inputs):
        if self.pool:
            inputs = self.pool_layer(inputs)
        inputs = self.convBlock(inputs)
        return inputs


class UpConvBlock(nn.Module):

    def __init__(
        self, 
        input_dim, 
        output_dim,
        kernel_size=3,
        stride=1,
        padding=1, 
        use_bn=1,
        slim=False,
        convFT=0
    ):
        
        super(UpConvBlock, self).__init__()

        self.upconv_layer = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding)
        self.upconv_layer.apply(init_weights)
        self.conv_block = DilatedConvBlock(input_dim, output_dim, kernel_size, stride, padding, use_bn, slim=slim, convFT=convFT)

    def forward(self, right, left):
        
        right = nn.functional.interpolate(right, mode='nearest', scale_factor=2)
        right = self.upconv_layer(right)
        
        left_shape = left.size()
        right_shape = right.size()
        padding = (left_shape[3] - right_shape[3], 0, left_shape[2] - right_shape[2], 0)

        right_pad = nn.ConstantPad2d(padding, 0)
        right = right_pad(right)
        out = torch.cat([right, left], 1)
        out =  self.conv_block(out)

        return out

