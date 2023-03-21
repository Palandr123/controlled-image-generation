from typing import Optional
import logging
import os

import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from gans import GanWrapper
from utils import LossMetric


class Trainer:
    """
    Model trainer
    :param self.manipulator - manipulator to train
    :param self.loss_fn - loss function
    :param self.optimizer - optimizer
    :param self.generator - pretrained generator
    :param self.embedder - embedder (either pretrained or not)
    :param self.device - device to the manipulator
    :param self.batch_size - number of batch elements
    :param self.iterations - number of iterations
    :param self.scheduler - learning rate scheduler (no scheduler if None)
    :param self.writer - writer logging metrics TensorBoard (disabled if None)
    :param self.save_path - directory in which checkpoints will be saved (no saving if None)
    :param self.checkpoint_path - checkpoint path, to resume training (no loading from checkpoint if None)
    """
    def __init__(self, manipulator: nn.Module, embedder: nn.Module, generator: GanWrapper, loss_fn: nn.Module,
                 optimizer: optim.Optimizer, batch_size: int, iterations: int, device: torch.device,
                 eval_freq: int = 1000, eval_iters: int = 100,
                 scheduler: Optional[optim.lr_scheduler.LRScheduler] = None, writer: Optional[SummaryWriter] = None,
                 save_path: Optional[str] = None, checkpoint_path: Optional[str] = None) -> None:

        self.logger = logging.getLogger()
        self.writer = writer
        self.save_path = save_path

        self.device = device
        self.manipulator = manipulator
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.generator = generator
        self.embedder = embedder

        self.eval_freq = eval_freq
        self.eval_iters = eval_iters
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.iterations = iterations
        self.start_iteration = 0

        if checkpoint_path:
            self._load_from_checkpoint(checkpoint_path)

        self.train_acc_metric = LossMetric()
        self.train_loss_metric = LossMetric()

        self.val_acc_metric = LossMetric()
        self.val_loss_metric = LossMetric()

        self.best_loss = None

    def train(self) -> None:
        """
        Train the model
        :return: None
        """
        self.logger.info("Beginning training")
        epoch = 0
        iteration = self.start_iteration
        while iteration < self.iterations:
            if iteration + self.eval_freq < self.iterations:
                num_iters = self.eval_freq
            else:
                num_iters = self.iterations - iteration

            self._train_loop(epoch, num_iters)
            self._val_loop(epoch, self.eval_iters)

            epoch_str = f"Epoch {epoch} "
            epoch_str += f"| Training accuracy: {self.train_acc_metric.compute():.3f} "
            epoch_str += f"| Training loss: {self.train_loss_metric.compute():.3f} "
            epoch_str += f"| Validation acc: {self.val_acc_metric.compute():.3f} "
            epoch_str += f"| Validation loss: {self.val_loss_metric.compute():.3f} "
            self.logger.info(epoch_str)

            if self.writer is not None:
                self.writer.add_scalar("Loss/train", self.train_loss_metric.compute(), epoch)
                self.writer.add_scalar("Acc/train", self.train_acc_metric.compute(), epoch)
                self.writer.add_scalar("Loss/val", self.val_loss_metric.compute(), epoch)
                self.writer.add_scalar("Acc/val", self.val_acc_metric.compute(), epoch)

            if self.save_path is not None:
                self._save_model(os.path.join(self.save_path, "last.pt"), iteration)

            val_loss = self.val_loss_metric.compute()
            if self.best_loss is None or val_loss < self.best_loss:
                self.best_loss = val_loss
                self._save_model(os.path.join(self.save_path, "best.pt"), iteration)

            self.train_loss_metric.reset()
            self.train_acc_metric.reset()
            self.val_loss_metric.reset()
            self.val_acc_metric.reset()

            iteration += num_iters
            epoch += 1
        self.logger.info("Finished training!")
        self._save_model(os.path.join(self.save_path, "final_model.pt"), self.iterations)

    def _train_loop(self, epoch: int, iterations: int) -> None:
        """
        Training epoch
        :param epoch: current epoch
        :param iterations: number of iterations
        :return: None
        """
        pbar = tqdm.tqdm(total=iterations, leave=False)
        pbar.set_description(f"Epoch {epoch} | Train")

        self.manipulator.train()
        self.generator.eval()
        self.embedder.train()

        for i in range(iterations):
            loss = self._iteration(pbar, "train")
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

        pbar.close()

    def _val_loop(self, epoch: int, iterations: int) -> None:
        """
        Validation epoch
        :param epoch: current epoch
        :param iterations: number of iterations
        :return: None
        """
        pbar = tqdm.tqdm(total=iterations, leave=False)
        pbar.set_description(f"Epoch {epoch} | Validation")

        self.manipulator.eval()
        self.generator.eval()
        self.embedder.eval()

        for i in range(iterations):
            with torch.no_grad():
                self._iteration(pbar, "val")

        pbar.close()

    def _iteration(self, pbar: tqdm.tqdm, stage: str = "train") -> torch.Tensor:
        """
        Iteration (either training or validation)
        :param pbar: progress bar to log the metrics
        :param stage: whether it is training and validation
        :return: Loss
        """
        z = torch.randn([self.batch_size, self.generator.z_dim])
        z = z.to(self.device)

        img_orig = self.generator(z)
        z_transformed = self.manipulator(z)

        features = []
        for j in range(z_transformed.shape[0] // self.batch_size):
            z_transformed_batch = z_transformed[j * self.batch_size:(j + 1) * self.batch_size]
            img_transformed = self.generator(z_transformed_batch)
            feats = self.embedder(img_orig, img_transformed)
            feats = feats / torch.reshape(torch.norm(feats, dim=1), (-1, 1))
            features.append(feats)
        features = torch.cat(features, dim=0)

        acc, loss = self.loss_fn(features)
        if stage == "train":
            self.train_acc_metric.update(acc.item(), z.shape[0])
            self.train_loss_metric.update(loss.item(), z.shape[0])
        else:
            self.val_acc_metric.update(acc.item(), z.shape[0])
            self.val_loss_metric.update(loss.item(), z.shape[0])

        pbar.update()
        pbar.set_postfix_str(
            f"Accuracy: {acc.item():.3f} Loss: {loss.item():.3f}", refresh=False
        )
        return loss

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

    def _save_model(self, path: str, iteration: int) -> None:
        """
        Save the checkpoint
        :param path: path to save the checkpoint
        :param iteration: iteration number
        :return: None
        """
        obj = {
            "iteration": iteration + 1,
            "optimizer": self.optimizer.state_dict(),
            "model": self.manipulator.state_dict(),
            "projector": self.embedder.state_dict(),
            "scheduler": self.scheduler.state_dict()
            if self.scheduler is not None else None,
        }
        torch.save(obj, path)
