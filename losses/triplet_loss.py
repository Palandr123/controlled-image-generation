import random

import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    """
    Triplet Loss class
    :param self.k - number of transformations
    :param self.p - the norm degree for pairwise distance
    :param margin - margin that will be added to the distance difference
    """
    def __init__(self, k: int, margin: float = 1, p: int = 2) -> None:
        super().__init__()
        self.k = k
        self.p = p
        self.triplet_loss = nn.TripletMarginLoss(margin, p)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        forward method for the loss
        :param features: extracted features, torch.tensor of the shape (k*batch_size, feature_dim), expected that the
                         batch is ordered by the transformation, i.e. the first **batch_size** elements in the first
                         dimension correspond to the first transformation
        :return: accuracy and loss
        """
        batch_size = features.size(0)
        num = batch_size // self.k
        loss = 0.0
        correct = 0
        for i in range(batch_size):
            num_transform = i // num
            idx_pos = random.choice([idx for idx in range(num_transform * num, (num_transform + 1) * num) if idx != i])
            idx_neg = random.choice([idx for idx in range(batch_size) if idx // num != num_transform])
            loss += self.triplet_loss(features[i], features[idx_pos], features[idx_neg])
            with torch.no_grad():
                correct += ((features[i] - features[idx_pos])**self.p).sum(axis=0).sqrt() < \
                           ((features[i] - features[idx_neg])**self.p).sum(axis=0).sqrt()
        return correct / batch_size, loss / batch_size
