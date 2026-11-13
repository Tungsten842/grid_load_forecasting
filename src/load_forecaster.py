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
    def __init__(self, input_length, d_model):
        super().__init__()
        self.embed = nn.Linear(input_length, d_model)

    def forward(self, x):
        return self.embed(x)


class TGM(nn.Module):
    def __init__(self, d_model, hidden_size):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
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
    def __init__(self, num_vars, hidden_size):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(num_vars, hidden_size),
            nn.ReLU(),
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
        input_length=INPUT_WINDOW,
        output_length=OUTPUT_WINDOW,
        embed_size=512,
        num_vars=3,
        num_quantiles=3,
    ):
        super().__init__()
        self.output_length = output_length

        self.revin = RevIN()

        self.embed = EMBED(input_length, embed_size)
        self.quantiles = num_quantiles

        self.glob1 = nn.Parameter(torch.randn(1, 1, embed_size))

        self.tgm = TGM(embed_size * 2, embed_size * 2)
        self.vgm = VGM(num_vars, embed_size * 2)

        self.head = nn.Linear(embed_size * 2, OUTPUT_WINDOW * num_quantiles)

    def forward(self, x):
        endo1, mean, std = self.revin.normalize(x[:, :, 0:1])
        exo1 = self.revin.normalize(x[:, :, 1:])[0]

        endo1 = self.embed(endo1.transpose(1, 2))
        exo1 = self.embed(exo1.transpose(1, 2))

        glob1 = self.glob1.expand_as(endo1)

        endo2, glob2 = self.tgm(endo1, glob1)

        glob3 = self.vgm(exo1, glob2)[1]

        y = self.head(torch.cat([endo2, glob3], -1))

        y = self.revin.denormalize(y, mean, std)

        y = y.reshape(-1, self.output_length, self.quantiles)
        return y
