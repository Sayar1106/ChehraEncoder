from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import datasets, transforms
from tqdm import tqdm

from agents.autoencoder import AutoEncoder


def create_model_architecture(
    train_example,
    enc_conv_f,
    enc_conv_ks,
    enc_conv_s,
    enc_p,
    dec_conv_t_f,
    dec_conv_t_ks,
    dec_conv_t_s,
    dec_p,
    dec_out_p,
    z_dim,
):

    return AutoEncoder(
        x_0=train_example,
        encoder_conv_filters=enc_conv_f,
        encoder_conv_kernel_size=enc_conv_ks,
        encoder_conv_strides=enc_conv_s,
        encoder_padding=enc_p,
        decoder_conv_t_filters=dec_conv_t_f,
        decoder_conv_t_kernel_size=dec_conv_t_ks,
        decoder_conv_t_strides=dec_conv_t_s,
        decoder_padding=dec_p,
        decoder_output_padding=dec_out_p,
        z_dim=z_dim,
    )


def train_autoencoder(
    batch_size=128,
    device="cuda",
    learning_rate=1e-5,
    betas=(0.9, 0.99),
    weight_decay=1e-2,
):
    train_data = datasets.MNIST(
        root="./data/", train=True, transform=transforms.ToTensor(), download=True
    )
    dataloader = DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True
    )

    model = create_model_architecture(
        training_example=train_data[0][0][None],
        enc_conv_f=[32, 64, 64, 64],
        enc_conv_ks=[3, 3, 3, 3],
        enc_conv_s=[1, 2, 2, 1],
        enc_p=[1, 1, 0, 1],
        dec_conv_t_f=[64, 64, 64, 32, 1],
        dec_conv_t_ks=[3, 3, 3, 3],
        dec_conv_t_s=[1, 2, 2, 1],
        dec_p=[1, 0, 1, 1],
        dec_out_p=[0, 1, 1, 0],
        z_dim=2,
    )
    model.cuda(device)

    optimizer = Adam(
        model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay
    )

    model.train()

    for epoch in tqdm(range(20)):
        for i, (data, _) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = F.mse_loss(pred, data)
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print(f"Loss: {loss}")

    return model
