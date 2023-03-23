import torch
import torch.nn as nn
from torchvision.models import resnet18

from utils import create_mlp


class Embedder(nn.Module):
    """
    class for embedding the transformations
    @:param self
    @:param self.features_extractor - ResNet-based network that produces the embeddings for the transformations
    """
    def __init__(self, n_layers: int, middle_features: int, out_features: int) -> None:
        super().__init__()
        self.feature_extractor = resnet18(weights='IMAGENET1K_V1')
        self.feature_extractor.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        in_features = self.feature_extractor.layer4[1].conv2.out_channels
        self.feature_extractor.fc = create_mlp(n_layers, in_features, middle_features, out_features)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        forward method for embedder
        :param img1: batch of images, torch.tensor of shape (N, z_dim)
        :param img2: batch of images that are results of the manipulation on latent vectors corresponding to img1
        :return: embedding for the manipulations, of the shape (N, out_features)
        """
        return self.feature_extractor(torch.cat([img1, img2], dim=1)).reshape(img1.size(0), -1)
