from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import datasets, transforms
from tqdm import tqdm

from agents.autoencoder import AutoEncoder


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

    model = AutoEncoder(
        x_0=train_data[0][0][None],
        input_dim=(28, 28, 1),
        encoder_conv_filters=[32, 64, 64, 64],
        encoder_conv_kernel_size=[3, 3, 3, 3],
        encoder_conv_strides=[1, 2, 2, 1],
        encoder_padding=[1, 1, 0, 1],
        decoder_conv_t_filters=[64, 64, 64, 32, 1],
        decoder_conv_t_kernel_size=[3, 3, 3, 3],
        decoder_conv_t_strides=[1, 2, 2, 1],
        decoder_padding=[1, 0, 1, 1],
        decoder_output_padding=[0, 1, 1, 0],
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