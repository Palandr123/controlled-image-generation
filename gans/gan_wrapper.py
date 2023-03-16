import torch
import torch.nn as nn


class GanWrapper(nn.Module):
    def forward(self, z: torch.FloatTensor) -> torch.FloatTensor:
        """
        forward method for GAN
        :param z: latent vector, torch.tensor of shape (N, z_dim)
        :return: generated images, torch.tensor of shape (N, img_height, img_width)
        """
        pass
