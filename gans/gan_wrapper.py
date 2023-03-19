import torch
import torch.nn as nn


class GanWrapper(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        forward method for GAN
        :param z: latent vector, torch.tensor of shape (N, z_dim)
        :return: generated images, torch.tensor of shape (N, img_height, img_width)
        """
        pass
