"""Hand-written LSTM-family cells built on ``nn.Linear``.

Three cells are implemented exactly as specified in the project appendix:

* ``VanillaLSTMCell`` - baseline LSTM (equations 1-6)
* ``CIFGCell``        - Coupled Input-Forget Gate LSTM (equations 13-18)
* ``WMCCell``         - Working Memory Connections LSTM (equations 7-12)

PyTorch's built-in recurrent modules are NOT used anywhere. The only
linear operation is ``nn.Linear``.

Implementation note (fused projections)
---------------------------------------
For speed, the per-gate matrices are stored side by side in one ``nn.Linear``:

* ``x_lin`` : stacks ``W_x*`` (one block per gate) and holds the biases ``b_*``.
* ``h_lin`` : stacks ``W_h*`` (one block per gate), without bias.

This is mathematically identical to using one ``nn.Linear`` per gate; the
output is simply split with ``chunk`` afterwards. The gate order inside the
fused matrices is documented in each class.

Every cell exposes two entry points:

* ``forward(x_t, state)``      - one time step from raw input ``x_t``.
* ``project_input`` + ``step`` - lets a recurrent layer compute ``W_x x_t + b``
  for the whole sequence in one matrix product, then loop over time.
"""

from typing import Dict, Optional, Tuple, Type

import torch
import torch.nn as nn

State = Tuple[torch.Tensor, torch.Tensor]  # (h, c)


class BaseLSTMCell(nn.Module):
    """Shared plumbing for all cells (fused nn.Linear projections, state init)."""

    n_gates: int = 4            # number of gate blocks stored in x_lin / h_lin
    forget_gate_index: int = 1  # position of the forget gate among those blocks

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # [W_x* | b_*] for all gates, and W_h* for all gates (no second bias).
        self.x_lin = nn.Linear(input_size, self.n_gates * hidden_size, bias=True)
        self.h_lin = nn.Linear(hidden_size, self.n_gates * hidden_size, bias=False)

    # ------------------------------------------------------------------ utils
    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> State:
        """Return a zero ``(h_0, c_0)`` pair."""
        ref = self.x_lin.weight
        h = torch.zeros(
            batch_size,
            self.hidden_size,
            device=device if device is not None else ref.device,
            dtype=dtype if dtype is not None else ref.dtype,
        )
        return h, h.clone()

    def set_forget_bias(self, value: float) -> None:
        """Set the bias block of the forget gate to ``value`` (common LSTM trick)."""
        hs, k = self.hidden_size, self.forget_gate_index
        with torch.no_grad():
            self.x_lin.bias[k * hs:(k + 1) * hs].fill_(value)

    def project_input(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``W_x x + b`` for all gates. ``x`` may be (B, D) or (B, T, D)."""
        return self.x_lin(x)

    # ------------------------------------------------------------ to override
    def step(self, x_proj: torch.Tensor, state: State) -> State:
        """One time step given the pre-computed input projection."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor, state: Optional[State] = None) -> State:
        """One time step from the raw input ``x`` of shape (B, input_size)."""
        if state is None:
            state = self.init_state(x.size(0), x.device, x.dtype)
        return self.step(self.project_input(x), state)

    def extra_repr(self) -> str:
        return f"input_size={self.input_size}, hidden_size={self.hidden_size}"


class VanillaLSTMCell(BaseLSTMCell):
    """Vanilla LSTM. Gate order in the fused matrices: [i, f, g, o].

    Trainable parameters: 4*H*(D+H) + 4*H.
    """

    n_gates = 4
    forget_gate_index = 1

    def step(self, x_proj: torch.Tensor, state: State) -> State:
        h_prev, c_prev = state
        # pre-activations: W_x* x_t + b_* (x_proj) + W_h* h_{t-1} (h_lin)
        pre = x_proj + self.h_lin(h_prev)
        pre_i, pre_f, pre_g, pre_o = pre.chunk(4, dim=-1)

        # i_t = sigmoid(W_hi h_{t-1} + W_xi x_t + b_i)          (1)
        i_t = torch.sigmoid(pre_i)
        # f_t = sigmoid(W_hf h_{t-1} + W_xf x_t + b_f)          (2)
        f_t = torch.sigmoid(pre_f)
        # o_t = sigmoid(W_ho h_{t-1} + W_xo x_t + b_o)          (3)
        o_t = torch.sigmoid(pre_o)
        # g_t = tanh(W_hg h_{t-1} + W_xg x_t + b_g)             (4)
        g_t = torch.tanh(pre_g)
        # c_t = f_t * c_{t-1} + i_t * g_t                       (5)
        c_t = f_t * c_prev + i_t * g_t
        # h_t = o_t * tanh(c_t)                                 (6)
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class CIFGCell(BaseLSTMCell):
    """Coupled Input-Forget Gate LSTM. Gate order in the fused matrices: [f, g, o].

    There is no input-gate matrix: i_t = 1 - f_t.
    Trainable parameters: 3*H*(D+H) + 3*H.
    """

    n_gates = 3
    forget_gate_index = 0

    def step(self, x_proj: torch.Tensor, state: State) -> State:
        h_prev, c_prev = state
        pre = x_proj + self.h_lin(h_prev)
        pre_f, pre_g, pre_o = pre.chunk(3, dim=-1)

        # f_t = sigmoid(W_hf h_{t-1} + W_xf x_t + b_f)          (13)
        f_t = torch.sigmoid(pre_f)
        # i_t = 1 - f_t                                         (14)
        i_t = 1.0 - f_t
        # o_t = sigmoid(W_ho h_{t-1} + W_xo x_t + b_o)          (15)
        o_t = torch.sigmoid(pre_o)
        # g_t = tanh(W_hg h_{t-1} + W_xg x_t + b_g)             (16)
        g_t = torch.tanh(pre_g)
        # c_t = f_t * c_{t-1} + i_t * g_t                       (17)
        c_t = f_t * c_prev + i_t * g_t
        # h_t = o_t * tanh(c_t)                                 (18)
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class WMCCell(BaseLSTMCell):
    """Working Memory Connections LSTM. Gate order in the fused matrices: [i, f, g, o].

    On top of the vanilla projections, the cell state feeds the gates through
    three extra ``nn.Linear`` (full H x H matrices, no bias):
    ``c_i`` = W_ci, ``c_f`` = W_cf, ``c_o`` = W_co.
    The output gate looks at the NEW cell state c_t, so c_t is computed first.

    Trainable parameters: 4*H*(D+H) + 4*H + 3*H*H.
    """

    n_gates = 4
    forget_gate_index = 1

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__(input_size, hidden_size)
        self.c_i = nn.Linear(hidden_size, hidden_size, bias=False)  # W_ci
        self.c_f = nn.Linear(hidden_size, hidden_size, bias=False)  # W_cf
        self.c_o = nn.Linear(hidden_size, hidden_size, bias=False)  # W_co

    def step(self, x_proj: torch.Tensor, state: State) -> State:
        h_prev, c_prev = state
        pre = x_proj + self.h_lin(h_prev)
        pre_i, pre_f, pre_g, pre_o = pre.chunk(4, dim=-1)

        # i_t = sigmoid(W_xi x_t + W_hi h_{t-1} + tanh(W_ci c_{t-1}) + b_i)   (7)
        i_t = torch.sigmoid(pre_i + torch.tanh(self.c_i(c_prev)))
        # f_t = sigmoid(W_xf x_t + W_hf h_{t-1} + tanh(W_cf c_{t-1}) + b_f)   (8)
        f_t = torch.sigmoid(pre_f + torch.tanh(self.c_f(c_prev)))
        # g_t = tanh(W_hg h_{t-1} + W_xg x_t + b_g)                           (10)
        g_t = torch.tanh(pre_g)
        # c_t = f_t * c_{t-1} + i_t * g_t                                     (11)
        # (computed BEFORE o_t because the output gate uses c_t)
        c_t = f_t * c_prev + i_t * g_t
        # o_t = sigmoid(W_xo x_t + W_ho h_{t-1} + tanh(W_co c_t) + b_o)       (9)
        o_t = torch.sigmoid(pre_o + torch.tanh(self.c_o(c_t)))
        # h_t = o_t * tanh(c_t)                                               (12)
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


CELL_REGISTRY: Dict[str, Type[BaseLSTMCell]] = {
    "lstm": VanillaLSTMCell,
    "cifg": CIFGCell,
    "wmc": WMCCell,
}

CELL_LABELS: Dict[str, str] = {
    "lstm": "Vanilla LSTM",
    "cifg": "CIFG",
    "wmc": "WMC",
}


def build_cell(name: str, input_size: int, hidden_size: int) -> BaseLSTMCell:
    """Instantiate a cell by name (``lstm``, ``cifg`` or ``wmc``)."""
    key = name.lower()
    if key not in CELL_REGISTRY:
        raise ValueError(f"Unknown cell '{name}'. Choose from {sorted(CELL_REGISTRY)}.")
    return CELL_REGISTRY[key](input_size, hidden_size)
