import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.conv import Conv2d
from torch.optim import Adam
from torchvision import datasets, transforms


class AutoEncoder(nn.Module):
    def __init__(
        self,
        x_0,
        input_dim,
        encoder_conv_filters,
        encoder_conv_kernel_size,
        encoder_conv_strides,
        decoder_conv_t_filters,
        decoder_conv_t_kernel_size,
        decoder_conv_t_strides,
        z_dim,
        use_batch_norm=False,
        use_dropout=False,
    ):

        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides

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
                    ),
                    nn.LeakyReLU(),
                ]
            )
            encoder_layers.append(nn.Sequential(*encoder_layer))
        x = nn.Sequential(*encoder_layers)(x_0)
        shape_pre_flatten = x.shape
        x = nn.Flatten()(x)

        encoder_layers.append(nn.Flatten())
        encoder_layers.append(nn.Linear(x.shape[1], self.z_dim))


        for i in range(1, len(self.decoder_conv_t_strides)):
            decoder_layer = []
            decoder_layer.extend(
                [
                    nn.ConvTranspose2d(
                        in_channels=self.decoder_conv_t_filters[i - 1],
                        out_channels=self.decoder_conv_t_filters[i],
                        kernel_size=self.decoder_conv_t_kernel_size[i],
                        stride=self.decoder_conv_t_strides[i]
                    ),
                    nn.Sigmoid() if i == len(self.decoder_conv_t_filters) - 2 else nn.LeakyReLU()
                ]
            )
            decoder_layers.append(nn.Sequential(*decoder_layer))

        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.fc1 = nn.Linear(self.z_dim, np.prod(shape_pre_flatten[1:]))
        self.decoder_layers = nn.Sequential(*decoder_layers)

    def forward(self, x):

        # x = self.decoder_input_layer(x)
        x = self.encoder_layers(x)
        x = self.fc1()(x)
        x = x.view(x.shape[0], *x.shape[1:])

        return self.decoder_layer(x)


if __name__ == "__main__":
    train_ds = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    model = AutoEncoder(
        x_0=train_ds[0][0][None],
        input_dim=(28, 28, 1),
        encoder_conv_filters=[32, 64, 64, 64],
        encoder_conv_kernel_size=[3, 3, 3, 3],
        encoder_conv_strides=[1, 2, 2, 1],
        decoder_conv_t_filters=[64, 64, 32, 1],
        decoder_conv_t_kernel_size=[3, 3, 3, 3],
        decoder_conv_t_strides=[1, 2, 2, 1],
        z_dim=2,
    )

    print(model)
