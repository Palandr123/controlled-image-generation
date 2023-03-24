import pickle

import torch

from gans.gan_wrapper import GanWrapper


class StyleGAN(GanWrapper):
    def __init__(self, weights_path: str) -> None:
        super().__init__()
        with open(weights_path, 'rb') as f:
            self.G = pickle.load(f)['G_ema']
        self.z_dim = self.G.z_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.G(z, None)

    def get_features(self, z: torch.Tensor) -> torch.Tensor:
        w = self.G.mapping(z, None)
        x = img = None
        layers = list(self.G.synthesis.children())
        idx = 0
        for i, layer in enumerate(layers[:7]):
            num = layer.num_conv + layer.num_torgb
            x, img = layer(x, img, w[:, idx:idx+num])
            idx = idx + layer.num_conv
        return img

    def sample_latent(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randn([batch_size, self.z_dim]).to(device)
