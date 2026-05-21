import torch

from .ml import AccentModel, MeterModel


class AccentModelONNX(AccentModel):
    def forward(self, accent_input, meter_class, pos_input):
        mask = (accent_input != -1).any(dim=-1)
        lengths = mask.sum(dim=1)
        _, T, _ = accent_input.shape

        accent_input = accent_input.masked_fill(~mask.unsqueeze(-1), 0.0)

        meter_emb = self.meter_emb(meter_class)
        meter_emb = meter_emb.unsqueeze(1).expand(-1, T, -1)

        pos_emb = self.pos_emb(pos_input + 1)

        x = torch.cat([accent_input, meter_emb, pos_emb], dim=-1)

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        out, _ = self.encoder(packed)

        out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=T,
        )

        logits = self.head(out).squeeze(-1)

        return logits


class MeterModelONNX(MeterModel):
    def forward(self, accent_input, pos_input):
        mask = (accent_input != -1).any(dim=-1)
        lengths = mask.sum(dim=1)
        _, T, _ = accent_input.shape

        x = accent_input.masked_fill(~mask.unsqueeze(-1), 0.0)

        pos_emb = self.pos_emb(pos_input + 1)

        x = torch.cat([x, pos_emb], dim=-1)

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        out, _ = self.encoder(packed)

        out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=T,
        )

        scores = self.attn(out).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)

        weights = torch.softmax(scores, dim=1)
        pooled = (out * weights.unsqueeze(-1)).sum(dim=1)

        return self.fc(pooled)
