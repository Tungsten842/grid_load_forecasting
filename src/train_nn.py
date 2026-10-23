from itertools import islice, pairwise

import mlflow
import numpy as np
import polars as pl
import torch
from line_profiler import profile
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm


class PinballLoss(nn.Module):
    def __init__(self, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(quantiles).view(1, 1, -1))

    def forward(self, y_pred, y_true):
        error = y_true.unsqueeze(-1) - y_pred
        return torch.maximum(
            self.quantiles * error,
            (self.quantiles - 1) * error,
        ).mean()


class RandomWindowDataset(IterableDataset):
    INPUT_WINDOW = 672
    OUTPUT_WINDOW = 336
    WINDOW_SIZE = INPUT_WINDOW + OUTPUT_WINDOW
    CHUNK_ROWS = 4194304
    SERIES_COL = "mean_consumption_per_meter"
    FEEDER_COL = "lv_feeder_unique_id"

    def __init__(self, path, batch_size, stride=48):
        super().__init__()
        self.stride = stride
        self.batch_size = batch_size

        self.lf = pl.scan_parquet(str(path)).select([self.FEEDER_COL, self.SERIES_COL])
        self.total_rows = self.lf.select(pl.len()).collect().item()

    @profile
    def __iter__(self):
        w_idx = torch.arange(self.WINDOW_SIZE)
        for offset in range(0, self.total_rows, self.CHUNK_ROWS):
            df = self.lf.slice(offset, self.CHUNK_ROWS).collect()

            if df.is_empty():
                continue

            vals = df[self.SERIES_COL].cast(pl.Float32).to_torch()

            feeder = df[self.FEEDER_COL].to_numpy()
            # Split different feeders
            boundaries = np.r_[
                0,
                np.flatnonzero(feeder[1:] != feeder[:-1]) + 1,
                feeder.size,
            ]

            valid_starts = [
                np.arange(start, stop - self.WINDOW_SIZE + 1, self.stride)
                for start, stop in pairwise(boundaries)
                if stop - start >= self.WINDOW_SIZE
            ]

            if not valid_starts:
                continue

            starts = np.concatenate(valid_starts)
            np.random.shuffle(starts)

            starts_t = torch.from_numpy(starts)
            for i in range(0, len(starts_t), self.batch_size):
                b = starts_t[i : i + self.batch_size, None]
                # Drop partial batches
                if len(b) < self.batch_size:
                    continue
                w = vals[b + w_idx]
                yield w[:, : self.INPUT_WINDOW], w[:, self.INPUT_WINDOW :]


class LoadForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(48 * 7 * 2, 336 * 3)
        self.l2 = nn.Linear(336 * 3, 336 * 3)
        self.l3 = nn.Linear(336 * 3, 336 * 3)
        self.l4 = nn.Linear(336 * 3, 336 * 3)
        self.act = nn.SiLU()

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-5
        x = (x - mean) / std

        y = self.l1(x)
        y = self.act(y)
        y = self.l2(y)
        y = self.act(y)
        y = self.l3(y)
        y = self.act(y)
        y = self.l4(y)

        y = y.reshape(-1, 336, 3)
        return y * std.unsqueeze(-1) + mean.unsqueeze(-1)


class LimitedIterableDataset(IterableDataset):
    def __init__(self, dataset, limit):
        self.dataset = dataset
        self.limit = limit

    def __iter__(self):
        yield from islice(iter(self.dataset), self.limit)


class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LoadForecaster().to(self.device)
        self.optimizer = Adam(self.model.parameters(), 0.00002)
        self.loss_fn = PinballLoss().to(self.device)

        train_dataset = RandomWindowDataset("dataset/train.parquet", 128)
        train_dataset = LimitedIterableDataset(train_dataset, 32768)

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=None,
            num_workers=0,
            persistent_workers=False,
            pin_memory=True,
        )

        val_dataset = RandomWindowDataset("dataset/val.parquet", 128, stride=48)

        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=None,
            num_workers=1,
            persistent_workers=True,
            pin_memory=True,
        )

    @profile
    def fit(self):
        pbar = tqdm(desc="train", unit="sample", unit_scale=True)

        with mlflow.start_run():
            running_train_loss = 0.0
            running_samples = 0
            total_samples = 0
            for batch in self.train_dataloader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                batch_size = inputs.size(0)
                pbar.update(batch_size)
                targets_pred = self.model(inputs)
                train_loss = self.loss_fn(targets_pred, targets)
                train_loss.backward()
                self.optimizer.step()

                running_train_loss += train_loss.item() * batch_size
                running_samples += batch_size
                total_samples += batch_size

                if total_samples % 32768 == 0 and total_samples > 0:
                    running_train_loss /= running_samples
                    mlflow.log_metric(
                        "train_loss", running_train_loss, step=total_samples
                    )
                    pbar.set_postfix(loss=f"{running_train_loss:.1f}", refresh=False)
                    running_train_loss = 0.0
                    running_samples = 0
                if total_samples % 524288 * 2 == 0 and total_samples > 0:
                    val_loss = self.evaluate()
                    mlflow.log_metric("val_loss", val_loss, step=total_samples)
            pbar.close()

    @profile
    def evaluate(self):
        with torch.inference_mode():
            self.model.eval()
            total_val_loss = 0.0
            val_pbar = tqdm(desc="val", unit="sample", unit_scale=True, leave=False)
            total_samples = 0
            for batch in self.val_dataloader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                batch_size = inputs.size(0)
                val_pbar.update(batch_size)
                targets_pred = self.model(inputs)
                test_loss = self.loss_fn(targets_pred, targets)
                total_val_loss += test_loss.item() * batch_size
                total_samples += batch_size

            val_pbar.close()
            val_loss = total_val_loss / total_samples
            self.model.train()
            return val_loss

    def save(self):
        torch.save(self.model.state_dict(), "model.pt")


def main():
    trainer = Trainer()
    trainer.fit()
    trainer.save()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
