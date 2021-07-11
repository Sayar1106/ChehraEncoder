import torch
import torch.nn.functional as F

def kl_loss(mu, log_var):
    return -0.5 * torch.sum(1 + log_var - torch.square(mu) - torch.exp(log_var), axis=1)


def vae_loss(data, pred, mu, log_var):
    return F.binary_cross_entropy(data, pred, reduction='sum') + kl_loss(mu, log_var)