import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from torchmetrics.functional import r2_score

class MLP(pl.LightningModule):

    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int, dropout_prob: int = 0.3):
        super().__init__()
        print(f"MLP with: layers of size: {input_size} -> {hidden_sizes} -> {output_size}.")
        assert len(hidden_sizes) > 0, "MLP must have at least 1 hidden layer!"

        self.input_size = input_size
        self.hidden_size = hidden_sizes
        self.output_size = output_size

        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.layer_norm = nn.LayerNorm(hidden_sizes[0])
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden = nn.Sequential()
        for i in range(len(hidden_sizes) - 1):
            self.hidden.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.hidden.append(nn.LayerNorm(hidden_sizes[i + 1]))
            self.hidden.append(nn.ReLU())
            self.hidden.append(nn.Dropout(dropout_prob))

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.dropout(self.layer_norm(self.input_layer(x))))
        x = self.hidden(x)
        x = self.output_layer(x)
        return x

    def common_step(self, batch, split: str):
        if len(batch) == 3:
            x, y, dataset = batch
            dataset_name = type(dataset).__name__
        elif len(batch) == 2:
            x, y = batch
        else:
            raise RuntimeError(f"Encountered abnormal batch of length {len(batch)}:\n {batch}")
        out = self(x)
        loss = F.mse_loss(out, y)
        r2 = r2_score(out, y)
        if len(batch) == 3:
            self.log(f"{dataset_name} {split} loss", loss, on_epoch=True, batch_size=len(x))
            self.log(f"{dataset_name} {split} R2", r2, on_epoch=True, batch_size=len(x))
        elif len(batch) == 2:
            self.log(f"{split} loss", loss, on_epoch=True, batch_size=len(x))
            self.log(f"{split} R2", r2, on_epoch=True, batch_size=len(x))
        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, "training")

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, "validation")

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, "test")

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer