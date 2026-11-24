import copy
import os

os.environ["POLARS_MAX_THREADS"] = "4"
import argparse
import time
from itertools import pairwise

import mlflow
import numpy as np
import polars as pl
import torch
from torch import nn
from torch.functional import F
from torch.optim import Adam
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from load_forecaster import INPUT_WINDOW, OUTPUT_WINDOW, TOTAL_WINDOW, LoadForecaster


class PinballLoss(nn.Module):
    def __init__(self, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.quantiles_list = quantiles
        self.register_buffer("quantiles", torch.tensor(quantiles))

    def forward(self, y_pred, y_true):
        error = y_true.unsqueeze(-1) - y_pred
        q_loss = torch.maximum(
            self.quantiles * error,
            (self.quantiles - 1) * error,
        )
        loss = q_loss.mean(dim=(0, 2))
        q_loss = q_loss.mean(dim=(0, 1))
        return loss, q_loss


class RandomWindowDataset(IterableDataset):
    FEEDER_COL = "lv_feeder_unique_id"

    def __init__(
        self, path, batch_size, n_meters=None, drop_last=False, CHUNK_ROWS=2**27
    ):
        super().__init__()
        self.CHUNK_ROWS = CHUNK_ROWS

        self.SERIES_COLS = ["mean_consumption_per_meter"]
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.lf = pl.scan_parquet(str(path)).select(
            [self.FEEDER_COL, *self.SERIES_COLS]
        )

        if n_meters:
            ids = (
                self.lf.select(self.FEEDER_COL)
                .unique()
                .sort(self.FEEDER_COL)
                .head(n_meters)
            )
            self.lf = self.lf.join(ids, on=self.FEEDER_COL, maintain_order="left")

        self.total_rows = self.lf.select(pl.len()).collect().item()

    def __iter__(self):
        w_idx = torch.arange(TOTAL_WINDOW)
        info = torch.utils.data.get_worker_info()
        start = 0 if not info else info.id * self.CHUNK_ROWS
        step = self.CHUNK_ROWS * (1 if not info else info.num_workers)
        for offset in range(start, self.total_rows, step):
            df = self.lf.slice(offset, self.CHUNK_ROWS).collect()

            if df.is_empty():
                continue

            vals = df.select(pl.col(self.SERIES_COLS).cast(pl.Float32)).to_torch()

            feeder = df[self.FEEDER_COL].to_numpy()
            # Split different feeders
            boundaries = np.r_[
                0,
                np.flatnonzero(feeder[1:] != feeder[:-1]) + 1,
                feeder.size,
            ]

            valid_starts = [
                np.arange(start, stop - TOTAL_WINDOW + 1, OUTPUT_WINDOW)
                for start, stop in pairwise(boundaries)
                if stop - start >= TOTAL_WINDOW
            ]

            if not valid_starts:
                continue

            starts = np.concatenate(valid_starts)
            np.random.shuffle(starts)

            bad = ~torch.isfinite(vals).all(dim=1)
            bad_before = F.pad(bad.cumsum(0), (1, 0))
            starts = torch.from_numpy(starts)
            starts_t = starts[bad_before[starts + TOTAL_WINDOW] == bad_before[starts]]

            for i in range(0, len(starts_t), self.batch_size):
                b = starts_t[i : i + self.batch_size, None]
                if self.drop_last and len(b) < self.batch_size:
                    continue
                w = vals[b + w_idx].to("cuda")
                yield w[:, :INPUT_WINDOW, :], w[:, -OUTPUT_WINDOW:, 0]


class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, epochs: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.epochs = epochs

    def __iter__(self):
        for _ in range(self.epochs):
            yield from super().__iter__()


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.best_model = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = copy.deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model):
        if self.best_model is not None:
            model.load_state_dict(self.best_model)


class Trainer:
    def __init__(self):
        torch.set_float32_matmul_precision("high")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.hyperparams = {
            # Model params
            "input_length": INPUT_WINDOW,
            "output_length": OUTPUT_WINDOW,
            "embed_size": 640,
            "num_quantiles": 3,
            "decoder_dim": 64,
            "temporal_decoder_hidden": 64,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            # Training params
            "learning_rate": 5e-4,
            "batch_size": 4096,
            "dropout": 0.00,
            "patience": 10,
        }

        self.model = LoadForecaster(
            input_length=self.hyperparams["input_length"],
            output_length=self.hyperparams["output_length"],
            num_quantiles=self.hyperparams["num_quantiles"],
            decoder_dim=self.hyperparams["decoder_dim"],
            num_encoder_layers=self.hyperparams["num_encoder_layers"],
            num_decoder_layers=self.hyperparams["num_decoder_layers"],
            temporal_decoder_hidden=self.hyperparams["temporal_decoder_hidden"],
            embed_size=self.hyperparams["embed_size"],
            dropout=self.hyperparams["dropout"],
        ).to(self.device)

        self.optimizer = Adam(
            self.model.parameters(), self.hyperparams["learning_rate"], fused=True
        )
        self.loss_fn = PinballLoss().to(self.device)

        self.dataloaders = {}

        train_dataset = RandomWindowDataset(
            "dataset/train.parquet", self.hyperparams["batch_size"], drop_last=True
        )
        self.dataloaders["train"] = InfiniteDataLoader(
            train_dataset,
            batch_size=None,
            num_workers=8,
            persistent_workers=True,
            epochs=1,
        )

        val_dataset = RandomWindowDataset(
            "dataset/val.parquet", self.hyperparams["batch_size"], CHUNK_ROWS=2**25
        )
        self.dataloaders["val"] = DataLoader(
            val_dataset,
            batch_size=None,
            num_workers=8,
            persistent_workers=True,
        )

        test_dataset = RandomWindowDataset(
            "dataset/test.parquet", self.hyperparams["batch_size"], CHUNK_ROWS=2**25
        )
        self.dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=None,
            num_workers=8,
        )

        test_unseen_dataset = RandomWindowDataset(
            "dataset/test_unseen.parquet", self.hyperparams["batch_size"]
        )
        self.dataloaders["test_unseen"] = DataLoader(
            test_unseen_dataset,
            batch_size=None,
            num_workers=8,
        )

    def fit(self):
        pbar = tqdm(desc="train", unit="sample", unit_scale=True)
        early_stopping = EarlyStopping(patience=self.hyperparams["patience"])

        with mlflow.start_run():
            mlflow.log_params(self.hyperparams)
            mlflow.log_artifacts("src/")
            running_train_loss = 0.0
            running_q_loss = 0.0
            running_samples = 0
            total_samples = 0
            t0 = time.perf_counter()

            for inputs, targets in self.dataloaders["train"]:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                batch_size = inputs.size(0)
                pbar.update(batch_size)
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    targets_pred = self.model(inputs)
                    train_loss_raw, q_loss = self.loss_fn(targets_pred, targets)

                train_loss = train_loss_raw.mean()
                train_loss.backward()
                self.optimizer.step()

                running_train_loss += train_loss.detach() * batch_size
                running_q_loss += q_loss.detach() * batch_size
                running_samples += batch_size
                total_samples += batch_size

                if total_samples % 2**19 == 0 and total_samples > 0:
                    running_train_loss /= running_samples
                    pbar.set_postfix(loss=f"{running_train_loss:.1f}", refresh=False)
                    samples_per_sec = running_samples / (time.perf_counter() - t0)
                    metrics = {
                        "train_pinball_loss_mean": running_train_loss,
                        "train_samples_per_sec": samples_per_sec,
                    }
                    for q, q_loss in zip(
                        self.loss_fn.quantiles_list,
                        (running_q_loss / running_samples).tolist(),
                    ):
                        metrics[f"train_pinball_loss_q{q}"] = q_loss

                    mlflow.log_metrics(metrics, step=total_samples)

                    running_train_loss = 0.0
                    running_q_loss = 0.0
                    running_samples = 0
                    t0 = time.perf_counter()
                if total_samples % 2**23 == 0 and total_samples > 0:
                    val_loss = self.evaluate("val", total_samples).item()
                    if early_stopping.step(val_loss, self.model):
                        break
                    t0 = time.perf_counter()
            pbar.close()

            val_loss = self.evaluate("val", total_samples).item()
            early_stopping.step(val_loss, self.model)
            early_stopping.restore_best(self.model)

            self.evaluate("test", total_samples)

            self.evaluate("test_unseen", total_samples)

    def evaluate(self, split, step):
        with torch.inference_mode():
            self.model.eval()
            total_val_loss = 0.0
            running_raw_loss = 0.0
            running_q_loss = 0.0
            val_pbar = tqdm(desc=split, unit="sample", unit_scale=True, leave=False)
            total_samples = 0
            for inputs, targets in self.dataloaders[split]:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                batch_size = inputs.size(0)
                val_pbar.update(batch_size)
                with (
                    torch.inference_mode(),
                    torch.autocast(device_type=self.device, dtype=torch.bfloat16),
                ):
                    targets_pred = self.model(inputs)
                    raw_loss, q_loss = self.loss_fn(targets_pred, targets)
                test_loss = raw_loss.mean()

                total_val_loss += test_loss.detach() * batch_size
                running_q_loss += q_loss.detach() * batch_size
                if split == "test":
                    running_raw_loss += raw_loss * batch_size
                total_samples += batch_size

            val_pbar.close()
            val_loss = total_val_loss / total_samples

            metrics = {f"{split}_pinball_loss_mean": val_loss}
            average_q_loss = running_q_loss / total_samples
            for q, l in zip(self.loss_fn.quantiles_list, average_q_loss):
                metrics[f"{split}_pinball_loss_q{q}"] = l.item()
            if split == "train" or split == "val":
                mlflow.log_metrics(metrics, step)
            else:
                mlflow.log_metrics(metrics)

            if split == "test":
                average_raw_loss = running_raw_loss / total_samples
                for i, loss in enumerate(average_raw_loss.tolist()):
                    mlflow.log_metric("horizon_test_pinball_loss_mean", loss, i)

            self.model.train()
            return val_loss

    def save(self):
        mlflow.pytorch.log_model(
            name="model", pytorch_model=self.model, serialization_format="pickle"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", type=str)
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment_name)
    trainer = Trainer()
    trainer.fit()
    trainer.save()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
