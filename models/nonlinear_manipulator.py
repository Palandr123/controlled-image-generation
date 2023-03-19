from typing import Optional

import torch
import torch.nn as nn
import numpy as np

from utils import create_mlp


class NonlinearManipulator(nn.Module):
    """
    Class for non-linear manipulations in the latent space of GANs
    :param self.k - amount of transformation
    :param self.z_dim - dimensionality of the latent space
    :param self.nets - neural networks corresponding to each manipulation
    :param self.alpha_range - sampling range for manipulation strength if not given
    """
    def __init__(self, k: int, z_dim: int, n_layers: int, alpha_range: tuple) -> None:
        super().__init__()
        self.k = k
        self.z_dim = z_dim
        self.alpha_range = alpha_range
        self.nets = nn.ModuleList([create_mlp(n_layers, z_dim, z_dim, z_dim) for _ in range(k)])

    def forward(self, z: torch.Tensor, alpha: Optional[float] = None) -> torch.Tensor:
        """
        forward method for manipulation
        :param z: batch of latent vectors, torch.tensor of shape (N, z_dim)
        :param alpha: manipulations strength
        :return: manipulated latent vectors, torch.tensor of shape (K, N, z_dim)
        """
        z = z.reshape((1, -1, self.z_dim))
        z = z.repeat((self.k, 1, 1))

        dz = []
        for i in range(self.k):
            res_dz = self.nets[i](z[i])
            res_dz /= torch.norm(res_dz, dim=1).reshape((-1, 1))
            if alpha is not None:
                res_dz *= alpha
            else:
                res_dz *= np.random.uniform(self.alpha_range[0], self.alpha_range[1], size=1)[0]
            dz.append(res_dz)

        dz = torch.stack(dz)
        z = z + dz
        return z

    def forward_single(self, z: torch.Tensor, k: int, alpha: Optional[float] = None) -> torch.Tensor:
        """
        forward method for manipulation of a single latent vector
        :param z: latent vector, torch.tensor of shape (1, z_dim)
        :param k: transformation index
        :param alpha: manipulations strength
        :return: manipulated latent vector, torch.tensor of shape (1, z_dim)
        """
        dz = self.nets[k](z)
        dz /= torch.norm(dz, dim=1).reshape((-1, 1))
        if alpha is not None:
            dz *= alpha
        else:
            dz *= np.random.uniform(self.alpha_range[0], self.alpha_range[1], size=1)[0]
        return z + dz
