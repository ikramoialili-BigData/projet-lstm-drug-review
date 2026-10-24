"""Embedding + (stacked, optionally bidirectional) recurrent cells + output layer.

Padding is handled with a boolean mask: at padded positions the hidden and cell states are
frozen, so the final state of every review is the state after its LAST REAL token, and the
padded outputs are zeroed (mean pooling) or ignored (max pooling).
"""
from typing import Tuple

import torch
import torch.nn as nn

from .cells import BaseCell, build_cell

POOLING_MODES = ("last", "mean", "max")


def reverse_padded(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Reverse each sequence of x (batch, time, dim) inside its real length; padding stays in place.

    Applying it twice gives back the input.
    """
    batch, time = x.size(0), x.size(1)
    t = torch.arange(time, device=x.device).unsqueeze(0)            # (1, T)
    reversed_index = lengths.unsqueeze(1) - 1 - t                    # (B, T)
    index = torch.where(t < lengths.unsqueeze(1), reversed_index, t.expand(batch, time))
    return x.gather(1, index.unsqueeze(-1).expand(-1, -1, x.size(2)))


class RecurrentLayer(nn.Module):
    """One recurrent layer (one cell, or two cells for the bidirectional case)."""

    def __init__(self, cell_name: str, input_size: int, hidden_size: int,
                 bidirectional: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.forward_cell = build_cell(cell_name, input_size, hidden_size)
        self.backward_cell = build_cell(cell_name, input_size, hidden_size) if bidirectional else None

    @property
    def output_size(self) -> int:
        return self.hidden_size * (2 if self.bidirectional else 1)

    @staticmethod
    def _run(cell: BaseCell, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one cell over x (B, T, D). Returns outputs (B, T, H) and the last real hidden state (B, H)."""
        batch, time, _ = x.shape
        x_proj = cell.project_input(x)                 # all input projections in one matmul
        h, c = cell.init_state(batch, x)
        outputs = []
        for t in range(time):
            h_new, c_new = cell.step(x_proj[:, t], (h, c))
            keep = mask[:, t].unsqueeze(1)             # (B, 1) True where the token is real
            h = torch.where(keep, h_new, h)            # freeze the state on padding
            c = torch.where(keep, c_new, c)
            outputs.append(h_new * keep.to(h_new.dtype))   # zero output on padding
        return torch.stack(outputs, dim=1), h

    def forward(self, x: torch.Tensor, lengths: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out_f, h_f = self._run(self.forward_cell, x, mask)
        if not self.bidirectional:
            return out_f, h_f
        x_rev = reverse_padded(x, lengths)
        out_b, h_b = self._run(self.backward_cell, x_rev, mask)
        out_b = reverse_padded(out_b, lengths)         # align backward outputs with the original positions
        return torch.cat([out_f, out_b], dim=-1), torch.cat([h_f, h_b], dim=-1)

    def reset_parameters(self, scheme: str, forget_bias: float) -> None:
        self.forward_cell.reset_parameters(scheme, forget_bias)
        if self.backward_cell is not None:
            self.backward_cell.reset_parameters(scheme, forget_bias)


class RNNClassifier(nn.Module):
    """Text classifier: embedding -> stacked recurrent layers -> pooling -> dropout -> linear."""

    def __init__(self, vocab_size: int, n_classes: int, cell: str = "lstm", embed_dim: int = 128,
                 hidden_size: int = 128, num_layers: int = 1, bidirectional: bool = False,
                 pooling: str = "last", dropout: float = 0.3, pad_id: int = 0,
                 init: str = "xavier", forget_bias: float = 1.0):
        super().__init__()
        if pooling not in POOLING_MODES:
            raise ValueError(f"pooling must be one of {POOLING_MODES}, got '{pooling}'.")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        self.pooling = pooling
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        layers, size = [], embed_dim
        for _ in range(num_layers):
            layer = RecurrentLayer(cell, size, hidden_size, bidirectional)
            layers.append(layer)
            size = layer.output_size
        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(size, n_classes)
        for layer in self.layers:
            layer.reset_parameters(init, forget_bias)   # initialization scheme applies to the cells

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """ids: (B, T) int, lengths: (B,) int >= 1. Returns logits (B, n_classes)."""
        time = ids.size(1)
        mask = torch.arange(time, device=ids.device).unsqueeze(0) < lengths.unsqueeze(1)   # (B, T)
        x = self.dropout(self.embedding(ids))
        outputs, last_hidden = x, None
        for index, layer in enumerate(self.layers):
            outputs, last_hidden = layer(x, lengths, mask)
            x = self.dropout(outputs) if index < len(self.layers) - 1 else outputs
        if self.pooling == "last":
            features = last_hidden
        elif self.pooling == "mean":
            features = outputs.sum(dim=1) / lengths.unsqueeze(1).to(outputs.dtype)
        else:  # max over the real tokens only
            masked = outputs.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            features = masked.max(dim=1).values
        return self.classifier(self.dropout(features))

    def cell_parameters(self):
        """Parameters of the recurrent cells only (without embedding and output layer)."""
        return [p for p in self.layers.parameters() if p.requires_grad]
