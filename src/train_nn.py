import os

os.environ["POLARS_MAX_THREADS"] = "4"
import argparse
from itertools import chain, islice, pairwise

import mlflow
import numpy as np
import polars as pl
import torch
from torch import nn
from torch.functional import F
from torch.optim import Adam
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from load_forecaster import (
    INPUT_WINDOW,
    OUTPUT_WINDOW,
    TOTAL_WINDOW,
    LoadForecaster,
)


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
    CHUNK_ROWS = 2**27
    FEEDER_COL = "lv_feeder_unique_id"

    def __init__(self, path, batch_size, n_meters=None, drop_last=False):
        super().__init__()
        self.SERIES_COLS = ["mean_consumption_per_meter", "temperature", "humidity"]
        self.batch_size = batch_size
        self.macro_batch_size = batch_size * 8
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

            for i in range(0, len(starts_t), self.macro_batch_size):
                b = starts_t[i : i + self.macro_batch_size]
                n = len(b) // self.batch_size
                if (self.drop_last and len(b) < self.macro_batch_size) or n == 0:
                    continue

                w = vals[b[: n * self.batch_size, None] + w_idx]
                x = w[:, :INPUT_WINDOW].reshape(n, self.batch_size, INPUT_WINDOW, -1)
                y = w[:, -OUTPUT_WINDOW:, 0].reshape(n, self.batch_size, OUTPUT_WINDOW)
                yield x, y


class LimitedIterableDataset(IterableDataset):
    def __init__(self, dataset, limit):
        self.dataset = dataset
        self.limit = limit

    def __iter__(self):
        yield from islice(iter(self.dataset), self.limit)


class Trainer:
    def __init__(self):
        torch.set_float32_matmul_precision("high")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.hyperparams = {
            # Model params
            "input_length": INPUT_WINDOW,
            "output_length": OUTPUT_WINDOW,
            "tff_hidden": 1024,
            "cff_hidden": 64,
            "embed_size": 512,
            "num_vars": 3,
            "num_quantiles": 3,
            # Training params
            "learning_rate": 4e-4,
            "batch_size": 2048,
            "dropout": 0.05,
        }

        self.model = LoadForecaster(
            input_length=self.hyperparams["input_length"],
            output_length=self.hyperparams["output_length"],
            tff_hidden=self.hyperparams["tff_hidden"],
            cff_hidden=self.hyperparams["cff_hidden"],
            embed_size=self.hyperparams["embed_size"],
            num_vars=self.hyperparams["num_vars"],
            num_quantiles=self.hyperparams["num_quantiles"],
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
        train_dataset = LimitedIterableDataset(train_dataset, 64)
        self.dataloaders["train"] = DataLoader(
            train_dataset,
            batch_size=None,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )

        val_dataset = RandomWindowDataset(
            "dataset/val.parquet", self.hyperparams["batch_size"]
        )
        self.dataloaders["val"] = DataLoader(
            val_dataset,
            batch_size=None,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )

        test_dataset = RandomWindowDataset(
            "dataset/test.parquet", self.hyperparams["batch_size"]
        )
        self.dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=None,
            num_workers=4,
            pin_memory=True,
        )

        test_unseen_dataset = RandomWindowDataset("dataset/test_unseen.parquet", 128)
        self.dataloaders["test_unseen"] = DataLoader(
            test_unseen_dataset,
            batch_size=None,
            num_workers=4,
            pin_memory=True,
        )

    def fit(self):
        pbar = tqdm(desc="train", unit="sample", unit_scale=True)

        with mlflow.start_run():
            mlflow.log_params(self.hyperparams)
            mlflow.log_artifacts("src/")
            running_train_loss = 0.0
            running_q_loss = 0.0
            running_samples = 0
            total_samples = 0
            batches = chain.from_iterable(
                zip(
                    inputs_macro.to(self.device, non_blocking=True),
                    target_macro.to(self.device, non_blocking=True),
                )
                for inputs_macro, target_macro in self.dataloaders["train"]
            )

            for inputs, targets in batches:
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

                if total_samples % 65536 == 0 and total_samples > 0:
                    running_train_loss /= running_samples
                    pbar.set_postfix(loss=f"{running_train_loss:.1f}", refresh=False)
                    metrics = {"train_pinball_loss_mean": running_train_loss}
                    for q, l in zip(
                        self.loss_fn.quantiles_list,
                        (running_q_loss / running_samples).tolist(),
                    ):
                        metrics[f"train_pinball_loss_q{q}"] = l
                    mlflow.log_metrics(metrics, step=total_samples)

                    running_train_loss = 0.0
                    running_q_loss = 0.0
                    running_samples = 0
                if total_samples % 4194304 == 0 and total_samples > 0:
                    self.evaluate("val", total_samples)
            pbar.close()

            self.evaluate("test", total_samples)

            self.evaluate("test_unseen", total_samples)

    def evaluate(self, split, step):
        with torch.inference_mode():
            self.model.eval()
            total_val_loss = 0.0
            running_raw_loss = 0.0
            running_q_loss = 0.0
            val_pbar = tqdm(desc="val", unit="sample", unit_scale=True, leave=False)
            total_samples = 0
            batches = chain.from_iterable(
                zip(
                    inputs_macro.to(self.device, non_blocking=True),
                    target_macro.to(self.device, non_blocking=True),
                )
                for inputs_macro, target_macro in self.dataloaders[split]
            )
            for inputs, targets in batches:
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
