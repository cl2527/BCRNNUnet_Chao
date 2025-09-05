"""
    MoDL for Cardiac QSM data and multi_echo GRE brain data (kspace)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dc_blocks import *
from models.unet_blocks import *
from models.initialization import *
from models.resBlocks import *
from models.danet import daBlock, CAM_Module  
from models.fa import faBlockNew
from models.unet import *
from models.complex_unet import *
from models.straight_through_layers import *
from models.BCRNN import BCRNNlayer, Conv2dFT, MultiLevelBCRNNlayer
from models.complex_BCRNN import ComplexBCRNNlayer
from models.BCLSTM import BCLSTMlayer
from utils.data import *
from utils.operators import *
from torch.utils.checkpoint import checkpoint


class Resnet_with_DC(nn.Module):
    '''
        For Cardiac QSM data
    '''

    def __init__(
        self,
        input_channels,
        filter_channels,
        lambda_dll2, # initializing lambda_dll2
        K=1
    ):
        super(Resnet_with_DC, self).__init__()
        self.resnet_block = []
        layers = ResBlock(input_channels, filter_channels, use_norm=2)
        for layer in layers:
            self.resnet_block.append(layer)
        self.resnet_block = nn.Sequential(*self.resnet_block)
        self.resnet_block.apply(init_weights)
        self.K = K
        self.lambda_dll2 = nn.Parameter(torch.ones(1)*lambda_dll2, requires_grad=True)

    def forward(self, x, csms, masks):
        device = x.get_device()
        x_start = x
        # self.lambda_dll2 = self.lambda_dll2.to(device)
        A = backward_forward_CardiacQSM(csms, masks, self.lambda_dll2)
        Xs = []
        for i in range(self.K):
            x_block = self.resnet_block(x)
            x_block1 = x - x_block[:, 0:2, ...]
            rhs = x_start + self.lambda_dll2*x_block1
            dc_layer = DC_layer(A, rhs)
            x = dc_layer.CG_iter()
            Xs.append(x)
        return Xs[-1]


class Resnet_with_DC2(nn.Module):
    '''
        For multi_echo GRE brain data
    '''
    def __init__(
        self,
        input_channels,
        filter_channels,
        necho=10, # number of echos of input
        flag_convFT=0,  # flag to use conv2DFT layer
        flag_hidden=1, # BCRNN hidden feature recurrency
        flag_bn=2,  # flag to use group normalization: 0: no normalization, 2: use group normalization

    ):
        super(Resnet_with_DC2, self).__init__()
        self.necho = necho

        self.flag_hidden = flag_hidden
        self.flag_bn = flag_bn

        n_ch = 2  # number of channels
        nd = 5  # number of CRNN/BCRNN/CNN layers in each iteration
        if self.flag_padding == 0:
            nf = 64  # number of filters
        else:
            nf = 64
        ks = 3  # kernel size
        self.n_ch = n_ch
        self.nd = nd
        self.nf = nf
        self.ks = k
        self.bcrnn = BCRNNlayer(n_ch, nf, ks, flag_convFT, flag_bn, flag_hidden)
        self.denoiser = Unet(
            input_channels=nf,
            output_channels=n_ch,
            num_filters=[2**i for i in range(5, 8)],
            use_bn=flag_bn,
            use_deconv=1,
            skip_connect=False,
            slim=False,
            convFT=flag_convFT
        )


    def forward(self, kdatas, csms, csm_lowres, masks, flip, test=False, x_input=None):

        if x_input is not None:
            x = x_input
            self.necho_pred = self.necho_pred_value
            # print('Echo prediction')
        else:
            self.necho_pred = 0
            # print('No prediction')

            x = torch_channel_deconcate(x).permute(0, 1, 3, 4, 2)  # (n, 2, nx, ny, n_seq)
            x = x.contiguous()
            net = {}
            n_batch, n_ch, width, height, n_seq = x.size()
            size_h = [n_seq*n_batch, self.nf, width, height]
            if test:
                with torch.no_grad():
                    hid_init = Variable(torch.zeros(size_h)).cuda()
            else:
                hid_init = Variable(torch.zeros(size_h)).cuda()
            for j in range(self.nd-1):
                net['t0_x%d'%j]=hid_init

            Xs = []
           
            # update auxiliary variable x0
            x_ = x.permute(4, 0, 1, 2, 3) # (n_seq, n, 2, nx, ny)
            x_ = x_.contiguous()   
            # net['t%d_x0'%(i-1)] = net['t%d_x0'%(i-1)].view(n_seq, n_batch, self.nf, width, height)
            # net['t%d_x0'%i] = self.bcrnn(x_, net['t%d_x0'%(i-1)], test)
            x0 = self.bcrnn(x_, test)
            x0_ = torch_channel_concate(x0[None, ...].permute(0, 2, 1, 3, 4), self.necho).contiguous()
            rhs = x_start + self.lambda_dll2*x0_
            dc_layer = DC_layer_multiEcho(A, rhs, echo_cat=self.echo_cat, necho=self.necho,
                                          flag_precond=self.flag_precond, precond=self.precond)
            x = dc_layer.CG_iter()
            if self.echo_cat:
                x = torch_channel_concate(x, self.necho)
            if self.flag_temporal_pred:
                x_last_echos = self.densenet(x)
                Xs.append(torch.cat((x, x_last_echos), 1))
            else:     
                Xs.append(x)
            x = torch_channel_deconcate(x).permute(0, 2, 1, 3, 4).view(-1, n_ch, width, height) # (n_seq, 2, nx, ny)
            x = x[None, ...].permute(0, 2, 3, 4, 1).contiguous()
    
            x = self.denoiser(x)
            x_ = x_.view(-1, n_ch, width, height)

            return Xs


class DenseBlock(nn.Module):
    def __init__(
        self,
        input_channels,  # input_echos*2
        output_channels,  # output_echos*2
        filter_channels=32  # channel after each conv
    ):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, filter_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(input_channels+filter_channels, filter_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(input_channels+2*filter_channels, filter_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(input_channels+3*filter_channels, filter_channels, 3, padding=1)
        self.conv5 = nn.Conv2d(input_channels+4*filter_channels, filter_channels, 3, padding=1)
        self.conv_final = nn.Conv2d(filter_channels, output_channels, 1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x0):
        x1 = self.relu(self.conv1(x0))
        x2 = self.relu(self.conv2(torch.cat((x0, x1), 1)))
        x3 = self.relu(self.conv3(torch.cat((x0, x1, x2), 1)))
        x4 = self.relu(self.conv4(torch.cat((x0, x1, x2, x3), 1)))
        x5 = self.relu(self.conv5(torch.cat((x0, x1, x2, x3, x4), 1)))
        return self.conv_final(x5)
