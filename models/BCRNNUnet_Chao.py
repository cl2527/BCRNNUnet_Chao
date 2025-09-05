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


class BCRNNUnet(nn.Module):
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
        opt = None
    ):
        super(BCRNNUnet, self).__init__()
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
        """
        if dilated_flag:
            self.denoiser = DUnet(
                input_channels=nf,
                output_channels=n_ch,
                num_filters=[2**i for i in range(nfilters_1, nfilters_2)],
                use_bn=flag_bn,
                use_deconv=1,
                skip_connect=False,
                slim=False,
                convFT=flag_convFT
            )
        """
        if denoiser_model == 'CFNet':
            self.denoiser = CFNet(
                input_channel = n_ch*necho,
                output_channel = n_ch*necho,
            )
        else:
            if opt is not None and (opt['correction_type'] == 'trajectory_aware'):
                
                    self.denoiser = Unet(
                        input_channels=nf+260,
                        output_channels=n_ch,
                        num_filters=[2**i for i in range(nfilters_1, nfilters_2)],
                        use_bn=flag_bn,
                        use_deconv=0,
                        skip_connect=False,
                        slim=False,
                        convFT=flag_convFT
                    )
            else:
                if not Af_correction:
                    self.denoiser = Unet(
                        input_channels=nf,
                        output_channels=n_ch,
                        num_filters=[2**i for i in range(nfilters_1, nfilters_2)],
                        use_bn=flag_bn,
                        use_deconv=0,
                        skip_connect=False,
                        slim=False,
                        convFT=flag_convFT
                    )
                else:
                    self.denoiser = Unet(
                        input_channels=nf+2,
                        output_channels=n_ch,
                        num_filters=[2**i for i in range(nfilters_1, nfilters_2)],
                        use_bn=flag_bn,
                        use_deconv=0,
                        skip_connect=False,
                        slim=False,
                        convFT=flag_convFT
                    )
        self.Af_correction = Af_correction
        self.opt = opt

    def forward(self, x_input = None,  test = False):
        # x_input: size(n, 2, nx, ny, n_echo)

        x__ = x_input.permute(4, 0, 1, 2, 3).contiguous()   # (n_echo, n, 2, nx, ny)
        if (self.opt is not None) and (self.opt['correction_type'] == 'trajectory_aware'):
            x_ = x__[0:self.necho,:,:,:,:] # (n_echo, n, 2, nx, ny)
            noise = x__[-260:,:,0,:,:]# (260, n, 1, nx, ny)
            noise = noise.permute(1,0,2,3).unsqueeze(0) # (1, n, 260, nx, ny)
            nt, nb, _, width, height = x_.shape
            noise = noise.expand(nt, nb, 260, width, height).reshape(-1,260,width,height)
            x0 = self.bcrnn(x_, test)
            x0 = x0.view(-1, self.nf, width, height)
            x0 = torch.cat((x0, noise), 1)
        else:
            if self.Af_correction:
                x_ = x__[0:-1,:,:,:,:]  # (n_echo-1, n, 2, nx, ny)
                noise = x__[-1,:,:,:,:].unsqueeze(0) # (1,n, 2, nx, ny)
                nt, nb, _, width, height = x_.shape
                noise = noise.expand(nt, nb, 2, width, height).reshape(-1,2,width,height)
                x0 = self.bcrnn(x_, test)
                x0 = x0.view(-1, self.nf, width, height)
                x0 = torch.cat((x0, noise), 1)
            else:
                x_ = x__
                nt, nb, _, width, height = x_.shape
                x0 = self.bcrnn(x_, test)
                x0 = x0.view(-1, self.nf, width, height)
        
        x = self.denoiser(x0)
        x_ = x.view(nt, nb, self.n_ch, width, height).contiguous()   
        x_ = x_.permute(1,2,3,4,0) # (n, 2, nx, ny, n_echo)
        return x_

