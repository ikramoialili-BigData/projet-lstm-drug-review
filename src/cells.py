"""Hand-written recurrent cells built on nn.Linear: Vanilla LSTM, CIFG and WMC.

The built-in recurrent layers of PyTorch are NOT used. Every gate pre-activation is
computed with nn.Linear modules.

Implementation notes
--------------------
* For speed, the weight matrices of several gates are fused in ONE nn.Linear whose
  output is split with .chunk(). This is mathematically identical to using one
  separate matrix per gate (each chunk of the fused weight IS the matrix W_xi, W_xf, ...).
* The input-side projection W_x x_t + b is computed once for the whole sequence
  (`project_input`), then each time step only needs the recurrent part (`step`).
  `forward(x_t, state)` runs both parts for a single step and is used in the tests.
* Notation: sigma = sigmoid, (.) = element-wise product, W_h* act on h_{t-1},
  W_x* act on x_t, W_c* act on the cell state (WMC only).
"""
from typing import Tuple

import torch
import torch.nn as nn

State = Tuple[torch.Tensor, torch.Tensor]

INIT_SCHEMES = ("default", "xavier", "orthogonal")


class BaseCell(nn.Module):
    """Common machinery: shapes, state creation and weight initialization."""

    n_gates_x = 4          # number of blocks of the fused input projection
    forget_index = 1       # position of the forget gate inside the fused input projection

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

    # ----- API used by the recurrent layer ---------------------------------------------
    def project_input(self, x: torch.Tensor) -> torch.Tensor:
        """Input-side part W_x x + b for every time step at once. x: (..., input_size)."""
        return self.x_linear(x)

    def init_state(self, batch_size: int, like: torch.Tensor) -> State:
        zeros = like.new_zeros(batch_size, self.hidden_size)
        return zeros, zeros.clone()

    def step(self, x_proj: torch.Tensor, state: State) -> State:  # pragma: no cover
        raise NotImplementedError

    def forward(self, x: torch.Tensor, state: State = None) -> State:
        """One time step. x: (batch, input_size). Returns (h_t, c_t)."""
        if state is None:
            state = self.init_state(x.size(0), x)
        return self.step(self.project_input(x), state)

    # ----- initialization --------------------------------------------------------------
    def _weight_groups(self):
        """List of (kind, nn.Linear, number_of_gate_blocks); kind is 'input' or 'recurrent'."""
        raise NotImplementedError

    def reset_parameters(self, scheme: str = "xavier", forget_bias: float = 1.0) -> None:
        """Re-initialize the weights.

        scheme:
          'default'    -> keep the PyTorch default initialization of nn.Linear;
          'xavier'     -> Xavier-uniform on every gate block, zero biases;
          'orthogonal' -> orthogonal on every recurrent block (W_h*, W_c*), Xavier-uniform
                          on the input blocks (W_x*), zero biases.
        In all cases the bias of the forget gate is then set to `forget_bias`.
        """
        scheme = scheme.lower()
        if scheme not in INIT_SCHEMES:
            raise ValueError(f"Unknown init scheme '{scheme}'. Choose among {INIT_SCHEMES}.")
        hidden = self.hidden_size
        with torch.no_grad():
            if scheme != "default":
                for kind, linear, n_blocks in self._weight_groups():
                    for k in range(n_blocks):
                        block = linear.weight[k * hidden:(k + 1) * hidden]
                        if scheme == "orthogonal" and kind == "recurrent":
                            nn.init.orthogonal_(block)
                        else:
                            nn.init.xavier_uniform_(block)
                nn.init.zeros_(self.x_linear.bias)
            start = self.forget_index * hidden
            self.x_linear.bias[start:start + hidden] = forget_bias


class VanillaLSTMCell(BaseCell):
    """Vanilla LSTM (baseline)."""

    n_gates_x = 4
    forget_index = 1  # fused order: [i, f, o, g]

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__(input_size, hidden_size)
        # W_xi, W_xf, W_xo, W_xg (stacked) and the biases b_i, b_f, b_o, b_g
        self.x_linear = nn.Linear(input_size, 4 * hidden_size, bias=True)
        # W_hi, W_hf, W_ho, W_hg (stacked)
        self.h_linear = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.reset_parameters("default")

    def _weight_groups(self):
        return [("input", self.x_linear, 4), ("recurrent", self.h_linear, 4)]

    def step(self, x_proj: torch.Tensor, state: State) -> State:
        h_prev, c_prev = state
        pre = x_proj + self.h_linear(h_prev)
        pre_i, pre_f, pre_o, pre_g = pre.chunk(4, dim=-1)

        # (1) i_t = sigma(W_hi h_{t-1} + W_xi x_t + b_i)
        i_t = torch.sigmoid(pre_i)
        # (2) f_t = sigma(W_hf h_{t-1} + W_xf x_t + b_f)
        f_t = torch.sigmoid(pre_f)
        # (3) o_t = sigma(W_ho h_{t-1} + W_xo x_t + b_o)
        o_t = torch.sigmoid(pre_o)
        # (4) g_t = tanh(W_hg h_{t-1} + W_xg x_t + b_g)
        g_t = torch.tanh(pre_g)
        # (5) c_t = f_t (.) c_{t-1} + i_t (.) g_t
        c_t = f_t * c_prev + i_t * g_t
        # (6) h_t = o_t (.) tanh(c_t)
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class CIFGCell(BaseCell):
    """Coupled Input-Forget Gate LSTM: the input gate is tied to the forget gate (i = 1 - f)."""

    n_gates_x = 3
    forget_index = 0  # fused order: [f, o, g]

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__(input_size, hidden_size)
        # W_xf, W_xo, W_xg (stacked) and the biases b_f, b_o, b_g
        self.x_linear = nn.Linear(input_size, 3 * hidden_size, bias=True)
        # W_hf, W_ho, W_hg (stacked)
        self.h_linear = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.reset_parameters("default")

    def _weight_groups(self):
        return [("input", self.x_linear, 3), ("recurrent", self.h_linear, 3)]

    def step(self, x_proj: torch.Tensor, state: State) -> State:
        h_prev, c_prev = state
        pre = x_proj + self.h_linear(h_prev)
        pre_f, pre_o, pre_g = pre.chunk(3, dim=-1)

        # (13) f_t = sigma(W_hf h_{t-1} + W_xf x_t + b_f)
        f_t = torch.sigmoid(pre_f)
        # (14) i_t = 1 - f_t
        i_t = 1.0 - f_t
        # (15) o_t = sigma(W_ho h_{t-1} + W_xo x_t + b_o)
        o_t = torch.sigmoid(pre_o)
        # (16) g_t = tanh(W_hg h_{t-1} + W_xg x_t + b_g)
        g_t = torch.tanh(pre_g)
        # (17) c_t = f_t (.) c_{t-1} + i_t (.) g_t
        c_t = f_t * c_prev + i_t * g_t
        # (18) h_t = o_t (.) tanh(c_t)
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class WMCCell(BaseCell):
    """Working Memory Connections LSTM.

    The input and forget gates receive tanh(W_c c_{t-1}); the output gate receives
    tanh(W_co c_t), so c_t must be computed BEFORE o_t. W_ci, W_cf, W_co are full
    (hidden x hidden) matrices, exactly as written in the equations of the assignment.
    """

    n_gates_x = 4
    forget_index = 1  # fused order: [i, f, o, g]

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__(input_size, hidden_size)
        # W_xi, W_xf, W_xo, W_xg (stacked) and the biases b_i, b_f, b_o, b_g
        self.x_linear = nn.Linear(input_size, 4 * hidden_size, bias=True)
        # W_hi, W_hf, W_ho, W_hg (stacked)
        self.h_linear = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        # W_ci, W_cf (stacked): applied to the previous cell state c_{t-1}
        self.c_prev_linear = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        # W_co: applied to the CURRENT cell state c_t
        self.c_curr_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.reset_parameters("default")

    def _weight_groups(self):
        return [
            ("input", self.x_linear, 4),
            ("recurrent", self.h_linear, 4),
            ("recurrent", self.c_prev_linear, 2),
            ("recurrent", self.c_curr_linear, 1),
        ]

    def step(self, x_proj: torch.Tensor, state: State) -> State:
        h_prev, c_prev = state
        pre = x_proj + self.h_linear(h_prev)
        pre_i, pre_f, pre_o, pre_g = pre.chunk(4, dim=-1)
        peep_i, peep_f = self.c_prev_linear(c_prev).chunk(2, dim=-1)

        # (7) i_t = sigma(W_xi x_t + W_hi h_{t-1} + tanh(W_ci c_{t-1}) + b_i)
        i_t = torch.sigmoid(pre_i + torch.tanh(peep_i))
        # (8) f_t = sigma(W_xf x_t + W_hf h_{t-1} + tanh(W_cf c_{t-1}) + b_f)
        f_t = torch.sigmoid(pre_f + torch.tanh(peep_f))
        # (10) g_t = tanh(W_hg h_{t-1} + W_xg x_t + b_g)
        g_t = torch.tanh(pre_g)
        # (11) c_t = f_t (.) c_{t-1} + i_t (.) g_t      (computed BEFORE o_t)
        c_t = f_t * c_prev + i_t * g_t
        # (9) o_t = sigma(W_xo x_t + W_ho h_{t-1} + tanh(W_co c_t) + b_o)   (uses the new c_t)
        o_t = torch.sigmoid(pre_o + torch.tanh(self.c_curr_linear(c_t)))
        # (12) h_t = o_t (.) tanh(c_t)
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


CELLS = {"lstm": VanillaLSTMCell, "cifg": CIFGCell, "wmc": WMCCell}
CELL_NAMES = tuple(CELLS.keys())
CELL_LONG_NAMES = {"lstm": "Vanilla LSTM", "cifg": "CIFG", "wmc": "WMC"}


def build_cell(name: str, input_size: int, hidden_size: int) -> BaseCell:
    """Create a cell from its short name ('lstm', 'cifg' or 'wmc')."""
    key = name.lower()
    if key not in CELLS:
        raise ValueError(f"Unknown cell '{name}'. Choose among {CELL_NAMES}.")
    return CELLS[key](input_size, hidden_size)
