"""Hierarchical router for CPR: maps the base model's last-layer hidden state h_t
and the pooled question context c to an expert-call logit."""

import torch
import torch.nn as nn


class HierarchicalRouterNetwork(nn.Module):
    """Macro encoder over the pooled question context + token-level micro router."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.macro_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.micro_router = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, current_hidden: torch.Tensor, pooled_context: torch.Tensor):
        z_domain = self.macro_encoder(pooled_context)
        if current_hidden.dim() == 3:
            z_domain = z_domain.unsqueeze(1).expand(-1, current_hidden.size(1), -1)
        return self.micro_router(torch.cat([current_hidden, z_domain], dim=-1))


def get_hidden_size(model) -> int:
    """Hidden size of a causal LM (Gemma3 nests it under `text_config`)."""
    cfg = model.config
    return cfg.text_config.hidden_size if hasattr(cfg, "text_config") else cfg.hidden_size
