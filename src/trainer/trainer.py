import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm
import torchaudio

from src.base import BaseTrainer
from src.base.base_text_encoder import BaseTextEncoder
from src.logger.utils import plot_spectrogram_to_buf
from src.metric.utils import calc_cer, calc_wer
from src.utils import inf_loop, MetricTracker, synthesis


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        config,
        device,
        dataloaders,
        text_encoder,
        lr_scheduler=None,
        len_epoch=None,
        skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device)
        self.skip_oom = skip_oom
        self.text_encoder = text_encoder
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }
        self.lr_scheduler = lr_scheduler
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "mel_loss",
            "duration_loss",
            "pitch_loss",
            "energy_loss",
            "grad norm",
            *[m.name for m in self.metrics],
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            "mel_loss",
            "duration_loss",
            "pitch_loss",
            "energy_loss",
            *[m.name for m in self.metrics],
            writer=self.writer,
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in [
            "src_seq",
            "mel_target",
            "energy_target",
            "pitch_target",
            "length_target",
            "mel_pos",
            "src_pos",
        ]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, db in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            # for db in batch:
            try:
                db = self.process_batch(
                    db,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch,
                        self._progress(batch_idx),
                        (
                            db["mel_loss"]
                            + db["duration_loss"]
                            + db["pitch_loss"]
                            + db["energy_loss"]
                        ).item(),
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self.model.eval()
                self._log_predictions(is_train=True, **db)
                self.model.train()
                self._log_spectrogram(db["mel_output"].detach())
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()

        (
            batch["mel_output"],
            batch["duration_predicted"],
            batch["pitch_predicted"],
            batch["energy_predicted"],
        ) = self.model(**batch)

        (
            batch["mel_loss"],
            batch["duration_loss"],
            batch["pitch_loss"],
            batch["energy_loss"],
        ) = self.criterion(**batch)

        metrics.update("mel_loss", batch["mel_loss"].item())
        metrics.update("duration_loss", batch["duration_loss"].item())
        metrics.update("pitch_loss", batch["pitch_loss"].item())
        metrics.update("energy_loss", batch["energy_loss"].item())

        if is_train:
            (
                batch["mel_loss"]
                + batch["duration_loss"]
                + batch["pitch_loss"]
                # + batch["energy_loss"]
            ).backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        if is_train:
            for met in self.metrics:
                metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_predictions(is_train=False, **batch)
            self._log_spectrogram(batch["spectrogram"])

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]
        target_sr = self.config["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def _log_predictions(
        self,
        is_train,
        *args,
        **kwargs,
    ):
        for i, text in enumerate(
            [
                "Suck my big dick, bitch!",
                "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
                "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
                "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space",
            ]
        ):
            for duration_alpha, pitch_alpha, energy_alpha in [
                (1.0, 1.0, 1.0),
                (1.0, 1.0, 1.2),
                (1.0, 1.0, 0.8),
                (1.0, 1.2, 1.0),
                (1.0, 0.8, 1.0),
                (1.2, 1.0, 1.0),
                (0.8, 1.0, 1.0),
                (0.8, 0.8, 0.8),
                (1.2, 1.2, 1.2),
            ]:
                output_path = (
                    f"{i}_d{duration_alpha}_p{pitch_alpha}_e{energy_alpha}.wav"
                )
                synthesis.synthesis(
                    self.model,
                    text,
                    output_path,
                    self.device,
                    duration_alpha,
                    pitch_alpha,
                    energy_alpha,
                )
                self.writer.add_audio(
                    output_path,
                    self.load_audio(output_path),
                    sample_rate=self.config["sr"],
                )

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
