import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.conv import Conv2d
from torch.optim import Adam

class AutoEncoder(nn.Module):
    def __init__(self
    , input_dim
    , encoder_conv_filters
    , encoder_conv_kernel_size
    , encoder_conv_strides
    , decoder_conv_t_filters
    , decoder_conv_t_kernel_size
    , decoder_conv_t_strides
    , z_dim
    , use_batch_norm=False
    , use_dropout=False):

        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides

        self.z_dim = z_dim

        self.conv_layers = nn.ModuleList()

        for i in range(len(self.encoder_conv_filters)):
            if i == 0:
                input_channel = self.input_dim[-1]
            else:
                input_channel=self.encoder_conv_filters[-1]
            self.conv_layers.append(
                nn.Conv2d(input_channel,
                          out_channels=self.encoder_conv_filters[i],
                          kernel_size=self.encoder_conv_kernel_size,
                          stride=self.encoder_conv_strides
                )
            )
        
        self.output_layer = nn.Linear(self.encoder_conv_filters[-1], self.z_dim)
    
    def forward(self, features):
        x = None
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            x = nn.LeakyReLU()(x)
        x = self.output_layer(x)

        return x


if __name__ == '__main__':
    model = AutoEncoder(
input_dim = (28,28,1)
, encoder_conv_filters = [32,64,64, 64]
, encoder_conv_kernel_size = [3,3,3,3]
, encoder_conv_strides = [1,2,2,1]
, decoder_conv_t_filters = [64,64,32,1]
, decoder_conv_t_kernel_size = [3,3,3,3]
, decoder_conv_t_strides = [1,2,2,1]
, z_dim = 2)

    print(model)
