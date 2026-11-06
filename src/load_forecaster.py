import torch
from torch import nn

INPUT_WINDOW = 672
GAP_WINDOW = 48
OUTPUT_WINDOW = 48
TOTAL_WINDOW = INPUT_WINDOW + GAP_WINDOW + OUTPUT_WINDOW


class RevIN(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1), requires_grad=True)

    def normalize(self, x):
        mean = x.mean(dim=-1, keepdim=True).detach()
        std = x.std(dim=-1, keepdim=True).detach() + 1e-5

        x = self.alpha * ((x - mean) / std) + self.beta
        return x, mean, std

    def denormalize(self, y, mean, std):
        y = std * ((y - self.beta) / self.alpha) + mean
        return y


class EMBED(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(INPUT_WINDOW, d_model),
        )

    def forward(self, x):
        return self.embed(x)


class TGM(nn.Module):
    def __init__(self, d_model, t_ff):
        super().__init__()
        self.tgm = nn.Sequential(
            nn.Linear(d_model, t_ff),
            nn.ReLU(),
            nn.Linear(t_ff, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.tgm(x) * x
        return x


class LoadForecaster(nn.Module):
    def __init__(self, embed_size=1024):
        super().__init__()
        self.revin = RevIN()

        self.embed = EMBED(embed_size)
        self.tgm = TGM(embed_size, embed_size * 2)

        self.head = nn.Linear(embed_size, OUTPUT_WINDOW * 3)

    def forward(self, x):
        y, mean, std = self.revin.normalize(x)

        x_endo1 = self.embed(y)

        x_endo2 = self.tgm(x_endo1)

        y = self.head(x_endo2)

        y = self.revin.denormalize(y, mean, std)

        y = y.reshape(-1, OUTPUT_WINDOW, 3)
        return y
