"""Continuous-time RNN (CTRNN) model definition."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .activations import ActivationName, activation_torch


class CTRNN(nn.Module):
    """
    Continuous-time RNN (rate-based) with Euler step, optional noise and constraints.

    Update:
        v_t = (1 - alpha) * v_{t-1} + alpha * ( W_in x_t + W_rec f_{t-1} + b )
        f_t = act(v_t) + post_noise
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_size: int = 64,
        output_dim: int = 1,
        *,
        dt: float = 100.0,
        tau: float = 100.0,
        activation: ActivationName = "relu",
        preact_noise: float = 0.0,
        postact_noise: float = 0.0,
        use_dale: bool = False,
        ei_ratio: float = 0.8,
        no_self_connections: bool = True,
        scaling: float = 1.0,
        bias: bool = True,
        train_input_layer: bool = True,
        input_sparsity: float = 0.05,
        recurrent_init_std: Optional[float] = None,
        recurrent_bias_init: str = "zero",
        recurrent_bias_constant: float = 0.0,
    ):
        super().__init__()
        self.I, self.H, self.O = input_dim, hidden_size, output_dim
        self.dt = float(dt)
        self.tau = float(tau)
        self.alpha = float(dt) / float(tau)
        self.oneminusalpha = 1.0 - self.alpha
        self._activation_name = activation
        self.preact_noise = float(preact_noise)
        self.postact_noise = float(postact_noise)
        self.use_dale = bool(use_dale)
        self.no_self_connections = bool(no_self_connections)
        self._train_input_layer = bool(train_input_layer)
        self._input_sparsity = float(input_sparsity)
        self._recurrent_init_std = None if recurrent_init_std is None else float(recurrent_init_std)
        self._recurrent_bias_init = str(recurrent_bias_init)
        self._recurrent_bias_constant = float(recurrent_bias_constant)

        # layers
        self.input_layer = nn.Linear(self.I, self.H, bias=bias)
        self.hidden_layer = nn.Linear(self.H, self.H, bias=bias)
        self.readout_layer = nn.Linear(self.H, self.O, bias=bias)

        # inits
        nn.init.kaiming_uniform_(self.input_layer.weight, a=0.0)
        nn.init.zeros_(self.input_layer.bias)

        if self._recurrent_init_std is None:
            nn.init.kaiming_uniform_(self.hidden_layer.weight, a=0.0)
            self.hidden_layer.weight.data *= scaling
        else:
            nn.init.normal_(self.hidden_layer.weight, mean=0.0, std=self._recurrent_init_std)
        nn.init.zeros_(self.hidden_layer.bias)

        nn.init.kaiming_uniform_(self.readout_layer.weight, a=0.0)
        nn.init.zeros_(self.readout_layer.bias)

        # optionally freeze input projection (reservoir-style)
        if not self._train_input_layer:
            self._sparsify_and_freeze_input_layer()

        # Dale's Law
        if self.use_dale:
            n_exc = int(round(ei_ratio * self.H))
            sign = torch.cat([torch.ones(n_exc), -torch.ones(self.H - n_exc)]).view(1, -1)
            self.register_buffer("dale_sign", sign)
            with torch.no_grad():
                W = self.hidden_layer.weight.data
                self.hidden_layer.weight.data = W.abs() * self.dale_sign

        # remove self-connections
        if self.no_self_connections:
            with torch.no_grad():
                self.hidden_layer.weight.data.fill_diagonal_(0.0)

        self._initialize_recurrent_bias()

        # noise gate
        self.register_buffer("_noise_enabled", torch.tensor(1, dtype=torch.uint8))

    # utils
    def _sparsify_and_freeze_input_layer(self) -> None:
        with torch.no_grad():
            if 0.0 < self._input_sparsity < 1.0:
                weight = self.input_layer.weight.data
                mask = torch.rand_like(weight).lt(self._input_sparsity)
                weight.mul_(mask)
            if self.input_layer.bias is not None:
                self.input_layer.bias.zero_()
        for param in self.input_layer.parameters():
            param.requires_grad_(False)

    def _initialize_recurrent_bias(self) -> None:
        if self.hidden_layer.bias is None:
            return
        mode = self._recurrent_bias_init
        with torch.no_grad():
            if mode == "zero":
                self.hidden_layer.bias.zero_()
            elif mode == "constant":
                self.hidden_layer.bias.fill_(self._recurrent_bias_constant)
            elif mode == "cancel_shifted_tanh_offset":
                self.hidden_layer.bias.copy_(-0.5 * self.hidden_layer.weight.data.sum(dim=1))
            else:
                raise ValueError(
                    "recurrent_bias_init must be one of: zero, constant, cancel_shifted_tanh_offset"
                )

    def act(self, x: torch.Tensor) -> torch.Tensor:
        return activation_torch(x, self._activation_name)

    def enable_noise(self, enabled: bool = True):
        self._noise_enabled.fill_(1 if enabled else 0)

    def train(self, mode: bool = True):
        super().train(mode)
        self.enable_noise(mode)
        return self

    def eval(self):
        super().eval()
        self.enable_noise(False)
        return self

    # core
    def init_state(self, B: int, device=None):
        v0 = torch.zeros(B, self.H, device=device)
        fr0 = self.act(v0)
        return fr0, v0

    def step(self, fr_t: torch.Tensor, v_t: torch.Tensor, u_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # affine transforms
        w_in_u = self.input_layer(u_t)
        w_h_fr = self.hidden_layer(fr_t)

        # continuous-time Euler update
        v_t = self.oneminusalpha * v_t + self.alpha * (w_in_u + w_h_fr)

        # optional pre-activation noise
        if self.preact_noise > 0.0 and bool(self._noise_enabled.item()):
            v_t = v_t + self.alpha * torch.randn_like(v_t) * self.preact_noise

        # nonlinearity
        fr = self.act(v_t)

        # optional post-activation noise
        if self.postact_noise > 0.0 and bool(self._noise_enabled.item()):
            fr = fr + torch.randn_like(fr) * self.postact_noise

        return fr, v_t

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        inputs: (T,B,I) -> returns (logits: (T,B,O), hidden_states: (T,B,H))
        """
        T, B, _ = inputs.shape
        device = inputs.device
        fr, v = self.init_state(B, device)
        hs = []
        for t in range(T):
            fr, v = self.step(fr, v, inputs[t])
            hs.append(fr)
        hidden_seq = torch.stack(hs, dim=0)
        logits = self.readout_layer(hidden_seq)
        return logits, hidden_seq

    def forward_sequence(self, x):
        logits, _ = self.forward(x)
        return logits

    def hidden_sequence(self, x):
        _, h = self.forward(x)
        return h

    # save/load
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: str = "cpu"):
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state)
