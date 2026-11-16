import torch
from torch import nn

INPUT_WINDOW = 672
GAP_WINDOW = 48
OUTPUT_WINDOW = 48
TOTAL_WINDOW = INPUT_WINDOW + GAP_WINDOW + OUTPUT_WINDOW


class RevIN(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def normalize(self, x):
        mean = x.mean(dim=self.dim, keepdim=True).detach()

        var = torch.var(x, dim=self.dim, keepdim=True, unbiased=False)
        std = torch.sqrt(var + 1e-5).detach()

        x = (x - mean) / std
        return x, mean, std

    def denormalize(self, y, mean, std):
        y = std * y + mean
        return y


class EMBED(nn.Module):
    def __init__(self, input_length, d_model, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embed = nn.Linear(input_length, d_model)

    def forward(self, x):
        x = self.dropout(x)
        return self.embed(x)


class TGM(nn.Module):
    def __init__(self, d_model, hidden_size, dropout):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, d_model),
            nn.Sigmoid(),
        )

    def forward(self, endo, glob):
        endo_token = torch.cat([endo, glob], -1)
        endo_token = self.gate(endo_token) * endo_token

        endo_size = endo.shape[-1]
        endo = endo_token[..., :endo_size]
        glob = endo_token[..., endo_size:]
        return endo, glob


class VGM(nn.Module):
    def __init__(self, num_vars, hidden_size, dropout):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(num_vars, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_vars),
            nn.Sigmoid(),
        )

    def forward(self, exo, glob):
        exo_token = torch.cat([exo, glob], 1).transpose(1, 2)
        exo_token = self.gate(exo_token) * exo_token

        exo_token = exo_token.transpose(1, 2)
        exo = exo_token[:, :-1, :]
        glob = exo_token[:, -1:, :]
        return exo, glob


class LoadForecaster(nn.Module):
    def __init__(
        self,
        input_length,
        output_length,
        tff_hidden,
        cff_hidden,
        embed_size,
        num_vars,
        num_quantiles,
        dropout,
    ):
        super().__init__()
        self.output_length = output_length

        self.revin = RevIN()

        self.embed = EMBED(input_length, embed_size, dropout)
        self.quantiles = num_quantiles

        self.glob1 = nn.Parameter(torch.randn(1, 1, embed_size))

        self.tgm = TGM(embed_size * 2, tff_hidden, dropout)
        self.vgm = VGM(num_vars, cff_hidden, dropout)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_size * 2, output_length * num_quantiles),
        )

    def forward(self, x):
        endo1, mean, std = self.revin.normalize(x[:, :, 0:1])
        exo1 = self.revin.normalize(x[:, :, 1:])[0]

        combined = torch.cat([endo1, exo1], dim=2)
        combined = self.embed(combined.transpose(1, 2))
        endo1 = combined[:, 0:1, :]
        exo1 = combined[:, 1:, :]

        glob1 = self.glob1.expand_as(endo1)

        endo2, glob2 = self.tgm(endo1, glob1)

        glob3 = self.vgm(exo1, glob2)[1]

        y = self.head(torch.cat([endo2, glob3], -1))

        y = self.revin.denormalize(y, mean, std)

        y = y.reshape(-1, self.output_length, self.quantiles)
        return y


class ToyLoadForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(INPUT_WINDOW, 336 * 3)
        self.l2 = nn.Linear(336 * 3, 336 * 3)
        self.l3 = nn.Linear(336 * 3, 336 * 3)
        self.l4 = nn.Linear(336 * 3, OUTPUT_WINDOW * 3)
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

        y = y.reshape(-1, OUTPUT_WINDOW, 3)
        return y * std.unsqueeze(-1) + mean.unsqueeze(-1)
