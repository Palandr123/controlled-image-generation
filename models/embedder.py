import torch
import torch.nn as nn

from utils import create_mlp


class Embedder(nn.Module):
    """
    class for embedding the transformations
    @:param self.net - multi layer perceptron for that produces the embeddings for the transformations
    """
    def __init__(self, n_layers: int, in_features: int, middle_features: int, out_features: int) -> None:
        super().__init__()
        self.net = create_mlp(n_layers, in_features, middle_features, out_features)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        forward method for embedder
        :param z1: batch of latent vectors, torch.tensor of shape (N, z_dim)
        :param z2: batch of latent vectors that are results of the manipulation on latent vectors z1
        :return: embedding for the manipulations, of the shape (N, out_features)
        """
        z = torch.cat([z1, z2], dim=1)
        out = self.net(z)
        return out / torch.norm(out, dim=1)
