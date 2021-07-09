import numpy as np
import torch
import torch.distributions as d
import torch.nn as nn
from torch.nn.modules.conv import Conv2d
from torchvision import datasets, transforms


class VariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        x_0,
        encoder_conv_filters,
        encoder_conv_kernel_size,
        encoder_conv_strides,
        encoder_padding,
        decoder_conv_t_filters,
        decoder_conv_t_kernel_size,
        decoder_conv_t_strides,
        decoder_padding,
        decoder_output_padding,
        z_dim,
        use_batch_norm=False,
        use_dropout=False,
    ):

        super(AutoEncoder, self).__init__()
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.encoder_padding = encoder_padding
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides
        self.decoder_padding = decoder_padding
        self.decoder_output_padding = decoder_output_padding
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.norm_dist = d.Normal(0, 1)
        self.norm_dist.loc = self.norm_dist.loc.cuda()

        self.z_dim = z_dim

        encoder_layers, decoder_layers = [], []

        for i in range(len(self.encoder_conv_filters)):
            encoder_layer = []
            if i == 0:
                input_channel = self.input_dim[-1]
            else:
                input_channel = self.encoder_conv_filters[i - 1]
            encoder_layer.extend(
                [
                    nn.Conv2d(
                        input_channel,
                        out_channels=self.encoder_conv_filters[i],
                        kernel_size=self.encoder_conv_kernel_size[i],
                        stride=self.encoder_conv_strides[i],
                        padding=self.encoder_padding[i],
                    ),
                    nn.LeakyReLU(),
                ]
            )

            if self.use_batch_norm:
                encoder_layer.append(nn.BatchNorm2d(self.encoder_conv_filters[i]))
            if self.use_dropout:
                encoder_layer.append(nn.Dropout(0.25))

            encoder_layers.append(nn.Sequential(*encoder_layer))
        x = nn.Sequential(*encoder_layers)(x_0)
        self.shape_pre_flatten = x.shape
        x = nn.Flatten()(x)

        encoder_layers.append(nn.Flatten())
        self.mu = nn.Linear(x.shape[1], self.z_dim)
        self.log_var = nn.Linear(x.shape[1], self.z_dim)

        for i in range(1, len(self.decoder_conv_t_filters)):
            decoder_layer = []
            decoder_layer.extend(
                [
                    nn.ConvTranspose2d(
                        in_channels=self.decoder_conv_t_filters[i - 1],
                        out_channels=self.decoder_conv_t_filters[i],
                        kernel_size=self.decoder_conv_t_kernel_size[i - 1],
                        stride=self.decoder_conv_t_strides[i - 1],
                        padding=self.decoder_padding[i - 1],
                        output_padding=self.decoder_output_padding[i - 1],
                    ),
                ]
            )

            if i == len(self.decoder_conv_t_filters) - 1:
                decoder_layer.append(nn.Sigmoid())
            else:
                decoder_layer.append(nn.LeakyReLU())
                if self.use_batch_norm:
                    decoder_layer.append(nn.BatchNorm2d(self.decoder_conv_t_filters[i]))
                if self.use_dropout:
                    decoder_layer.append(nn.Dropout(0.25))
            decoder_layers.append(nn.Sequential(*decoder_layer))

        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.fc2 = nn.Linear(self.z_dim, np.prod(self.shape_pre_flatten[1:]))
        self.decoder_layers = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder_layers(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        epsilon = self.norm_dist.sample(mu.shape)
        z = mu + torch.exp(log_var / 2) * epsilon
        x = self.fc2(z)
        x = x.view(x.shape[0], *self.shape_pre_flatten[1:])

        return self.decoder_layers(x)
