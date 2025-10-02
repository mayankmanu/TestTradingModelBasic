from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyValueLSTM(nn.Module):
    """
    Minimal LSTM-based actor-critic.
    Input: feature vector (e.g., 5 dims from OHLCV normalization)
    Actor head -> logits over 3 actions (hold, buy, sell)
    Critic head -> scalar value
    Hidden state is maintained externally and can be saved/restored.
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, 3)
        self.critic = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size: int = 1, device: Optional[torch.device] = None):
        device = device or next(self.parameters()).device
        h0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return (h0, c0)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        # x shape: [B, T, input_dim] or [B, input_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B,1,F]
        out, hidden_out = self.lstm(x, hidden)
        last = out[:, -1, :]
        logits = self.actor(last)
        value = self.critic(last).squeeze(-1)
        return logits, value, hidden_out
