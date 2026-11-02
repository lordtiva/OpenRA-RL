"""Commander network v5: actor MLP trunk + spatial transformer + specialized heads + CTDE critic.

- item head (BUILD / TRAIN): logits over the item vocabulary + wait, conditioned on
  the queue, masked to what the queue offers right now (with dynamic economic sanity caps).
- tactical head (ARMY groups MAIN / HARASS / AIR + SUPPORT powers): contextualized
  spatial transformer over target candidates, scoring every target for both normal engagement
  (0..15) and fearless all_in assault (16..31, MAIN only), plus HOLD (32) and KEEP (33).
- ability head (DECIDE_ABILITY): unified 17-output pointer over the 16 target candidates
  and the CLS token (16 = HOLD), conditioned on `ability_emb`.
- macro_ops head (DECIDE_MACRO_OPS): 3-way classifier over ("repair", "sell", "hold").
- critic (training only, CTDE): an independent MLP over [g, priv] with zero shared
  weights with the actor trunk, preventing value-gradient interference on the policy.
"""

from typing import Iterator

import torch
from torch import nn

from ra_rl.features import (
    CAND_DIM,
    FEATURE_VERSION,
    GLOBAL_DIM,
    MAX_TARGETS,
    N_ARMY_ACTIONS,
    N_ITEM_ACTIONS,
    PRIV_DIM,
)
from ra_rl.vocab import MACRO_OPERATIONS, QUEUES, SPECIAL_ABILITIES, TACTICAL_CHANNELS

NEG = -1e9


class CommanderNet(nn.Module):
    def __init__(self, hidden: int = 256, cand_hidden: int = 128, priv_dim: int = PRIV_DIM):
        super().__init__()
        self.priv_dim = priv_dim
        self.config = {
            "hidden": hidden,
            "cand_hidden": cand_hidden,
            "priv_dim": priv_dim,
            "feature_version": FEATURE_VERSION,
        }
        self.trunk = nn.Sequential(
            nn.Linear(GLOBAL_DIM, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        )
        self.queue_emb = nn.Embedding(len(QUEUES), hidden)
        self.tactical_emb = nn.Embedding(len(TACTICAL_CHANNELS), hidden)
        self.ability_emb = nn.Embedding(len(SPECIAL_ABILITIES), hidden)

        # Contextualized spatial transformer (v5)
        self.cls_token_proj = nn.Linear(hidden, hidden)
        self.cand_proj = nn.Linear(CAND_DIM, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 2,
            batch_first=True,
            norm_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Output heads
        self.item_head = nn.Sequential(nn.ReLU(), nn.Linear(hidden, N_ITEM_ACTIONS))
        self.cand_score = nn.Sequential(
            nn.Linear(hidden, cand_hidden), nn.ReLU(),
            nn.Linear(cand_hidden, 2),
        )
        self.army_fixed = nn.Linear(hidden, 2)  # HOLD, KEEP
        self.ability_head = nn.Linear(hidden, 1)  # 17 tokens -> 17 logits (0..15 candidates, 16 = HOLD)
        self.macro_ops_head = nn.Linear(hidden, len(MACRO_OPERATIONS))  # repair, sell, hold

        # Independent asymmetric critic trunk + scalar head (CTDE; unused at eval time).
        self.critic = nn.Sequential(
            nn.Linear(GLOBAL_DIM + priv_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def actor_parameters(self) -> Iterator[nn.Parameter]:
        for name, param in self.named_parameters():
            if not name.startswith("critic."):
                yield param

    def critic_parameters(self) -> Iterator[nn.Parameter]:
        return self.critic.parameters()

    def reset_critic(self) -> None:
        for m in self.critic.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        last = self.critic[-1]
        if isinstance(last, nn.Linear):
            nn.init.orthogonal_(last.weight, gain=0.1)

    def value(self, g: torch.Tensor, priv: torch.Tensor | None = None) -> torch.Tensor:
        if priv is None:
            priv = torch.zeros(g.shape[0], self.priv_dim, device=g.device, dtype=g.dtype)
        return self.critic(torch.cat([g, priv], dim=-1)).squeeze(-1)

    def item_logits(self, g: torch.Tensor, queue: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.trunk(g) + self.queue_emb(queue)
        return self.item_head(h).masked_fill(~mask, NEG)

    def army_logits(
        self,
        g: torch.Tensor,
        cand: torch.Tensor,
        mask: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if group is None:
            group = torch.zeros(g.shape[0], device=g.device, dtype=torch.long)
        h = self.trunk(g) + self.tactical_emb(group)
        cls_tok = self.cls_token_proj(h).unsqueeze(1)                   # B, 1, H
        cand_toks = self.cand_proj(cand)                                # B, 16, H
        tokens = torch.cat([cand_toks, cls_tok], dim=1)                 # B, 17, H
        trans_tokens = self.spatial_transformer(tokens)                 # B, 17, H
        cand_ctx = trans_tokens[:, :MAX_TARGETS, :]                     # B, 16, H
        pair_scores = self.cand_score(cand_ctx)                         # B, 16, 2
        logits = torch.cat(
            [pair_scores[..., 0], pair_scores[..., 1], self.army_fixed(h)],
            dim=-1,
        )                                                               # B, 34
        assert logits.shape[-1] == N_ARMY_ACTIONS
        return logits.masked_fill(~mask, NEG)

    def ability_logits(
        self,
        g: torch.Tensor,
        cand: torch.Tensor,
        mask: torch.Tensor,
        ability_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if ability_id is None:
            ability_id = torch.zeros(g.shape[0], device=g.device, dtype=torch.long)
        h = self.trunk(g) + self.ability_emb(ability_id)
        cls_tok = self.cls_token_proj(h).unsqueeze(1)                   # B, 1, H
        cand_toks = self.cand_proj(cand)                                # B, 16, H
        tokens = torch.cat([cand_toks, cls_tok], dim=1)                 # B, 17, H
        trans_tokens = self.spatial_transformer(tokens)                 # B, 17, H
        logits = self.ability_head(trans_tokens).squeeze(-1)            # B, 17 (0..15 candidates, 16 = HOLD)
        return logits.masked_fill(~mask, NEG)

    def macro_ops_logits(
        self,
        g: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.trunk(g)
        logits = self.macro_ops_head(h)                                 # B, 3
        if mask is not None:
            logits = logits.masked_fill(~mask, NEG)
        return logits

    def save(self, path: str, **extra) -> None:
        torch.save({"config": self.config, "state": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "CommanderNet":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        if cfg.get("feature_version") != FEATURE_VERSION:
            raise ValueError(f"{path}: features v{cfg.get('feature_version')}, code is v{FEATURE_VERSION}")
        net = cls(
            hidden=cfg["hidden"],
            cand_hidden=cfg["cand_hidden"],
            priv_dim=cfg.get("priv_dim", PRIV_DIM),
        )
        missing, unexpected = net.load_state_dict(ckpt["state"], strict=False)
        non_critic_missing = [k for k in missing if not k.startswith("critic.")]
        unexpected = [k for k in unexpected if not k.startswith(("critic.", "value_head."))]
        if non_critic_missing or unexpected:
            raise RuntimeError(f"Error loading {path}: missing {non_critic_missing}, unexpected {unexpected}")
        return net.to(device).eval()
