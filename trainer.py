from typing import Optional
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from gans import GanWrapper


class Trainer:
    """
    Model trainer
    @:param self.manipulator - manipulator to train
    @:param self.loss_fn - loss function
    @:param self.optimizer - optimizer
    @:param self.generator - pretrained generator
    @:param self.embedder - embedder (either pretrained or not)
    @:param self.device - device to the manipulator
    @:param self.batch_size - number of batch elements
    @:param self.iterations - number of iterations
    @:param self.scheduler - learning rate scheduler (no scheduler if None)
    @:param self.writer - writer logging metrics TensorBoard (disabled if None)
    @:param self.save_path - directory in which checkpoints will be saved (no saving if None)
    @:param self.checkpoint_path - checkpoint path, to resume training (no loading from checkpoint if None)
    """
    def __init__(self, manipulator: nn.Module, loss_fn: nn.Module, optimizer: optim.Optimizer, generator: GanWrapper,
                 embedder: nn.Module, batch_size: int, iterations: int, device: torch.device, eval_freq: int = 1000,
                 eval_iters: int = 100, scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
                 writer: Optional[SummaryWriter] = None, save_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None, train_embedder: bool = True) -> None:

        self.logger = logging.getLogger()
        self.writer = writer
        self.save_path = save_path

        self.device = device
        self.manipulator = manipulator
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.generator = generator
        self.embedder = embedder

        self.train_embedder = train_embedder
        self.eval_freq = eval_freq
        self.eval_iters = eval_iters
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.iterations = iterations
        self.start_iteration = 0

        if checkpoint_path:
            self._load_from_checkpoint(checkpoint_path)

        self.best_loss = -1

    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load all the weights and parameters from the checkpoint
        :param checkpoint_path: path to the checkpoint, checkpoint should contain state dicts for the manipulator,
                                embedder, optimizer, scheduler (if necessary) and the iteration number
        :return: None
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.manipulator.load_state_dict(checkpoint["manipulator"])
        self.embedder.load_state_dict(checkpoint["embedder"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.start_iteration = checkpoint["iteration"]
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.logger.info(f"Successfully loaded the checkpoint, resuming from iteration {self.start_iteration}")
