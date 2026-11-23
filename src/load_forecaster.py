from torch import nn

INPUT_WINDOW = 672
GAP_WINDOW = 48
OUTPUT_WINDOW = 48
TOTAL_WINDOW = INPUT_WINDOW + GAP_WINDOW + OUTPUT_WINDOW


class ResBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, use_ln=True):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(input_dim, output_dim)
        self.ln = nn.LayerNorm(output_dim) if use_ln else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return self.ln(out + residual)


class TiDE(nn.Module):
    def __init__(
        self,
        input_length,
        output_length,
        num_quantiles,
        embed_size,
        decoder_dim,
        temporal_decoder_hidden,
        num_encoder_layers,
        num_decoder_layers,
        dropout,
    ):
        super().__init__()
        self.output_length = output_length
        self.decoder_dim = decoder_dim
        self.num_quantiles = num_quantiles

        encoder_layers = [
            ResBlock(
                input_length if i == 0 else embed_size,
                embed_size,
                embed_size,
                dropout,
            )
            for i in range(num_encoder_layers)
        ]
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_output_dim = output_length * decoder_dim
        decoder_layers = [
            ResBlock(
                embed_size,
                embed_size,
                decoder_output_dim if i == num_decoder_layers - 1 else embed_size,
                dropout,
            )
            for i in range(num_decoder_layers)
        ]
        self.decoder = nn.Sequential(*decoder_layers)

        self.temporal_decoder = ResBlock(
            input_dim=decoder_dim,
            hidden_dim=temporal_decoder_hidden,
            output_dim=num_quantiles,
            dropout=dropout,
            use_ln=False,
        )

        self.linear_skip = nn.Linear(input_length, output_length * num_quantiles)

    def forward(self, x):
        batch_size = x.size(0)

        endo = x[:, :, 0:1]

        mean = endo.mean(dim=1, keepdim=True).detach()
        std = endo.std(dim=1, keepdim=True).detach() + 1e-5
        endo = ((endo - mean) / std).squeeze(-1)

        e = self.encoder(endo)
        d = self.decoder(e)

        d = d.reshape(batch_size, self.output_length, self.decoder_dim)

        out_dense = self.temporal_decoder(d)
        skip = self.linear_skip(endo).reshape(
            batch_size, self.output_length, self.num_quantiles
        )

        out = out_dense + skip
        out = out * std + mean

        return out


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
