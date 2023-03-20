from pathlib import Path
import pickle

import torch

from gans.gan_wrapper import GanWrapper


class StyleGAN(GanWrapper):
    def __init__(self, weights_path: Path, device: str) -> None:
        super().__init__()
        self.device = torch.device(device)
        with open(str(weights_path), 'rb') as f:
            self.G = pickle.load(f)['G_ema'].cuda()
        self.z_dim = self.G.z_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.G(z, None)
