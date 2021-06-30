import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam

class AutoEncoder(nn.Module):
    def __init__(self
    , input_dim
    , encoder_conv_filters
    , encoder_kernel_size
    , encoder_conv_strides
    , decoder_conv_t_filters
    , decoder_conv_t_kernel_size
    , decoder_conv_t_strides
    , z_dim
    , use_batch_norm
    , use_dropout):

        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_kernel_size = encoder_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides

        self.z_dim = z_dim

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self._build()

    
    def _build(self):
        encoder_input = nn.
        


