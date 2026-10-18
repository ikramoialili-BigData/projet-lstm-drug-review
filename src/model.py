"""Text classifier: embedding -> (stacked, optionally bidirectional) recurrent cell -> linear head.

The recurrent part is always built from the hand-written cells of ``cells.py``
(vanilla LSTM, CIFG or WMC). Padding is handled with a mask: at padded time steps
the hidden and cell states are frozen, so the final state of each review is the
state at its last real token and the padding amount never changes the output.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .cells import BaseLSTMCell, build_cell

POOLING_CHOICES = ("last", "mean", "max")
INIT_CHOICES = ("xavier", "orthogonal", "default")


class RecurrentLayer(nn.Module):
    """Runs one cell over a padded batch, in forward or reverse time order."""

    def __init__(self, cell_type: str, input_size: int, hidden_size: int,
                 reverse: bool = False) -> None:
        super().__init__()
        self.cell = build_cell(cell_type, input_size, hidden_size)
        self.reverse = reverse

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, D) float, mask: (B, T) bool (True = real token).

        Returns all hidden states (B, T, H) and the final state (B, H).
        With right padding, the reverse direction starts (state frozen at zero)
        and only begins updating once it reaches the last real token.
        """
        batch, steps, _ = x.shape
        h, c = self.cell.init_state(batch, x.device, x.dtype)
        x_proj = self.cell.project_input(x)              # one big matmul for all steps
        mask_f = mask.unsqueeze(-1).to(x.dtype)          # (B, T, 1)
        order = range(steps - 1, -1, -1) if self.reverse else range(steps)

        outputs = [None] * steps
        for t in order:
            h_new, c_new = self.cell.step(x_proj[:, t], (h, c))
            m = mask_f[:, t]
            h = m * h_new + (1.0 - m) * h                # freeze state on padding
            c = m * c_new + (1.0 - m) * c
            outputs[t] = h
        return torch.stack(outputs, dim=1), h


class StackedRecurrent(nn.Module):
    """Several layers of the SAME cell type (still the same model family)."""

    def __init__(self, cell_type: str, input_size: int, hidden_size: int, num_layers: int = 1,
                 bidirectional: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.output_size = hidden_size * self.num_directions

        self.fwd_layers = nn.ModuleList()
        self.bwd_layers = nn.ModuleList()
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else self.output_size
            self.fwd_layers.append(RecurrentLayer(cell_type, in_size, hidden_size, reverse=False))
            if bidirectional:
                self.bwd_layers.append(RecurrentLayer(cell_type, in_size, hidden_size, reverse=True))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = x
        final = None
        for layer in range(self.num_layers):
            if layer > 0:
                out = self.dropout(out)
            out_f, h_f = self.fwd_layers[layer](out, mask)
            if self.bidirectional:
                out_b, h_b = self.bwd_layers[layer](out, mask)
                out = torch.cat([out_f, out_b], dim=-1)
                final = torch.cat([h_f, h_b], dim=-1)
            else:
                out, final = out_f, h_f
        return out, final


class TextClassifier(nn.Module):
    """Embedding + recurrent encoder + dropout + linear output layer."""

    def __init__(self, vocab_size: int, num_classes: int, cell_type: str = "lstm",
                 embed_dim: int = 128, hidden_size: int = 128, num_layers: int = 1,
                 bidirectional: bool = False, dropout: float = 0.3, pooling: str = "last",
                 pad_idx: int = 0) -> None:
        super().__init__()
        if pooling not in POOLING_CHOICES:
            raise ValueError(f"pooling must be one of {POOLING_CHOICES}")
        self.pad_idx = pad_idx
        self.pooling = pooling
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(dropout)
        self.rnn = StackedRecurrent(cell_type, embed_dim, hidden_size, num_layers,
                                    bidirectional, dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.rnn.output_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) token ids (long). Returns logits (B, num_classes)."""
        mask = x != self.pad_idx                                  # (B, T) True = real token
        emb = self.embed_dropout(self.embedding(x))
        out, final = self.rnn(emb, mask)

        if self.pooling == "last":
            features = final
        elif self.pooling == "mean":
            m = mask.unsqueeze(-1).to(out.dtype)
            features = (out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        else:  # max
            features = out.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
        return self.classifier(self.dropout(features))


def _init_blocks(weight: torch.Tensor, n_blocks: int, init_fn) -> None:
    """Apply ``init_fn`` to each gate block (row slice) of a fused weight matrix."""
    rows = weight.size(0) // n_blocks
    for k in range(n_blocks):
        init_fn(weight.data[k * rows:(k + 1) * rows])


def initialize_weights(model: TextClassifier, scheme: str = "xavier",
                       forget_bias: Optional[float] = 1.0) -> None:
    """Initialise the recurrent cells and the output layer.

    * ``default``    : keep PyTorch's default ``nn.Linear`` initialisation.
    * ``xavier``     : Xavier-uniform on every gate block, zero biases.
    * ``orthogonal`` : (semi-)orthogonal matrix for every gate block, zero biases.

    For ``xavier`` / ``orthogonal`` the forget-gate bias is then set to
    ``forget_bias`` (pass ``None`` to leave it at zero). The embedding keeps its
    default initialisation in all cases.
    """
    scheme = scheme.lower()
    if scheme not in INIT_CHOICES:
        raise ValueError(f"init must be one of {INIT_CHOICES}")
    if scheme == "default":
        return

    init_fn = nn.init.xavier_uniform_ if scheme == "xavier" else nn.init.orthogonal_
    for module in model.rnn.modules():
        if not isinstance(module, BaseLSTMCell):
            continue
        _init_blocks(module.x_lin.weight, module.n_gates, init_fn)
        _init_blocks(module.h_lin.weight, module.n_gates, init_fn)
        nn.init.zeros_(module.x_lin.bias)
        for name in ("c_i", "c_f", "c_o"):                 # WMC cell-state connections
            if hasattr(module, name):
                init_fn(getattr(module, name).weight.data)
        if forget_bias is not None:
            module.set_forget_bias(forget_bias)

    nn.init.xavier_uniform_(model.classifier.weight)
    nn.init.zeros_(model.classifier.bias)
