import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from gans import GanWrapper
from trainer import Trainer


@hydra.main(version_base=None, config_path="./configs", config_name="train.yaml")
def train(cfg: DictConfig) -> None:
    """
    Train the model
    :param cfg: config with params
    :return: None
    """
    device = torch.device(cfg.device if torch.cuda.is_available else "cpu")

    manipulator: nn.Module = hydra.utils.instantiate(cfg.manipulator, k=cfg.k).to(device)
    embedder: nn.Module = hydra.utils.instantiate(cfg.embedder).to(device)
    generator: GanWrapper = hydra.utils.instantiate(cfg.generator).to(device)
    loss_fn: nn.Module = hydra.utils.instantiate(cfg.loss, k=cfg.k).to(device)

    optimizer: torch.optim.Optimizer = hydra.utils.instantiate(cfg.hparams.optimizer,
                                                               list(manipulator.parameters()) +
                                                               list(embedder.parameters()))
    scheduler = hydra.utils.instantiate(cfg.hparams.scheduler, optimizer)

    save_path = cfg.save_path
    checkpoint_path = cfg.checkpoint_path

    if cfg.use_tensorboard:
        writer = SummaryWriter()
        text = f"<pre>{OmegaConf.to_yaml(cfg)}</pre>"
        writer.add_text("config", text)
    else:
        writer = None

    trainer = Trainer(manipulator, embedder, generator, loss_fn, optimizer, cfg.hparams.batch_size,
                      cfg.hparams.iterations, device, cfg.eval_freq, cfg.eval_iters, scheduler, writer, save_path,
                      checkpoint_path)
    trainer.train()


if __name__ == "__main__":
    train()
