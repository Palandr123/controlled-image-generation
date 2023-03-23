import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss class
    :param self.k - number of transformations
    :param self.temp - temperature scaling for the exponent
    :param self.reduction - whether sum or mean reduction are applied (mean is the default)
    """
    def __init__(self, k: int, temp: float, reduction: str = "mean") -> None:
        super().__init__()
        self.k = k
        self.temp = temp
        self.reduction = reduction

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        forward method for the loss
        :param features: extracted features, torch.tensor of the shape (k*batch_size, feature_dim), expected that the
                         batch is ordered by the transformation, i.e. the first **batch_size** elements in the first
                         dimension correspond to the first transformation
        :return: accuracy and loss
        """
        batch_size = features.size(0)

        similarity_matrix = torch.mm(features, features.t().contiguous())
        similarity_matrix = torch.abs(similarity_matrix)
        similarity_matrix = torch.exp(similarity_matrix * self.temp)

        mask = torch.zeros((batch_size, batch_size), device=similarity_matrix.device).bool()
        for i in range(self.k):
            start, end = i * (batch_size // self.k), (i + 1) * (batch_size // self.k)
            mask[start:end, start:end] = 1

        diag_mask = ~(torch.eye(batch_size, device=similarity_matrix.device).bool())

        pos = similarity_matrix.masked_select(mask * diag_mask).view(batch_size, -1)
        neg = similarity_matrix.masked_select(~mask).view(batch_size, -1)

        if self.reduction == "sum":
            pos = pos.sum(dim=-1)
            neg = neg.sum(dim=-1)
        else:
            pos = pos.mean(dim=-1)
            neg = neg.mean(dim=-1)

        acc = (pos > neg).float().mean()
        loss = -torch.log(pos / neg).mean()
        return acc, loss
