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
from models.dilated_unet import *
from models.complex_unet import *
from models.straight_through_layers import *
from models.BCRNN import BCRNNlayer, Conv2dFT, MultiLevelBCRNNlayer
from models.complex_BCRNN import ComplexBCRNNlayer
from models.BCLSTM import BCLSTMlayer
from models.CFNet import CFNet
from utils.data import *
from utils.operators import *


class BCRNNUnet_stacked(nn.Module):
    '''
        For multi_echo GRE brain data
    '''
    def __init__(
        self,
        input_channels,
        nfilters_1,
        nfilters_2,
        necho=5, # number of echos of input
        flag_convFT=0,  # flag to use conv2DFT layer
        flag_hidden=1, # BCRNN hidden feature recurrency
        flag_bn=2,  # flag to use group normalization: 0: no normalization, 2: use group normalization
        dilated_flag = 0,
        Af_correction = False,
        denoiser_model = 'Unet',
    ):
        super(BCRNNUnet_stacked, self).__init__()
        self.necho = necho
        self.flag_hidden = flag_hidden
        self.flag_bn = flag_bn

        n_ch = 2  # number of channels
        nd = necho  # number of CRNN/BCRNN/CNN layers in each iteration
        nf = input_channels
        ks = 3  # kernel size
        self.n_ch = n_ch
        self.nd = nd
        self.nf = nf
        self.bcrnn = BCRNNlayer(n_ch, nf, ks, flag_convFT, flag_bn, flag_hidden)

        self.denoiser0 = Unet(
            input_channels=nf,
            output_channels=n_ch,
            num_filters=[2**i for i in range(nfilters_1, nfilters_2)],
            use_bn=flag_bn,
            use_deconv=0,
            skip_connect=False,
            slim=False,
            convFT=flag_convFT
        )
        self.denoiser = Unet(
            input_channels=n_ch,
            output_channels=n_ch,
            num_filters=[2**i for i in range(nfilters_1, nfilters_2)],
            use_bn=flag_bn,
            use_deconv=0,
            skip_connect=False,
            slim=False,
            convFT=flag_convFT
        )
        self.Af_correction = Af_correction

    def forward(self, x_input = None,  test = False):
        # x_input: size(n, 2, nx, ny, n_echo)
        
        x_ = x_input.permute(4, 0, 1, 2, 3).contiguous()   # (n_echo, n, 2, nx, ny)

        nt, nb, _, width, height = x_.shape
        x = self.bcrnn(x_, test)
        x = x.view(-1, self.nf, width, height)

        x = self.denoiser0(x)
        
        n_iter = 2
        for i in range(n_iter):
            x = self.denoiser(x)
            if i == 0:
                x1 = x
        
        output = x.view(nt, nb, self.n_ch, width, height).contiguous()   
        output = output.permute(1,2,3,4,0) # (n, 2, nx, ny, n_echo)
        
        x1 = x1.view(nt, nb, self.n_ch, width, height).contiguous()
        x1 = x1.permute(1,2,3,4,0)
        return output, x1
