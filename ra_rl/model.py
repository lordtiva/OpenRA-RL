"""Commander network: a shared MLP trunk and two heads.

- item head (BUILD / TRAIN): logits over the item vocabulary + wait, conditioned on
  the queue, masked to what the queue offers right now.
- army head (ARMY): a pointer over the target candidates (each scored against the
  trunk) plus two fixed actions, hold and keep.
"""

import torch
from torch import nn

from ra_rl.features import CAND_DIM, FEATURE_VERSION, GLOBAL_DIM, MAX_TARGETS, N_ITEM_ACTIONS
from ra_rl.vocab import QUEUES

NEG = -1e9


class CommanderNet(nn.Module):
    def __init__(self, hidden: int = 256, cand_hidden: int = 128):
        super().__init__()
        self.config = {"hidden": hidden, "cand_hidden": cand_hidden, "feature_version": FEATURE_VERSION}
        self.trunk = nn.Sequential(
            nn.Linear(GLOBAL_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.queue_emb = nn.Embedding(len(QUEUES), hidden)
        self.item_head = nn.Sequential(nn.ReLU(), nn.Linear(hidden, N_ITEM_ACTIONS))
        self.cand_enc = nn.Sequential(nn.Linear(CAND_DIM, cand_hidden), nn.ReLU())
        self.cand_score = nn.Sequential(nn.Linear(hidden + cand_hidden, cand_hidden), nn.ReLU(),
                                        nn.Linear(cand_hidden, 1))
        self.army_fixed = nn.Linear(hidden, 2)  # hold, keep
        self.value_head = nn.Sequential(nn.ReLU(), nn.Linear(hidden, 1))

    def value(self, g: torch.Tensor) -> torch.Tensor:
        h = self.trunk(g)
        return self.value_head(h).squeeze(-1)

    def item_logits(self, g: torch.Tensor, queue: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.trunk(g) + self.queue_emb(queue)
        return self.item_head(h).masked_fill(~mask, NEG)

    def army_logits(self, g: torch.Tensor, cand: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.trunk(g)
        c = self.cand_enc(cand)                                            # B, T, C
        hc = torch.cat([h.unsqueeze(1).expand(-1, MAX_TARGETS, -1), c], dim=-1)
        scores = self.cand_score(hc).squeeze(-1)                           # B, T
        logits = torch.cat([scores, self.army_fixed(h)], dim=-1)           # B, T + 2
        return logits.masked_fill(~mask, NEG)

    def save(self, path: str, **extra) -> None:
        torch.save({"config": self.config, "state": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "CommanderNet":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        if cfg.get("feature_version") != FEATURE_VERSION:
            raise ValueError(f"{path}: features v{cfg.get('feature_version')}, code is v{FEATURE_VERSION}")
        net = cls(hidden=cfg["hidden"], cand_hidden=cfg["cand_hidden"])
        missing, unexpected = net.load_state_dict(ckpt["state"], strict=False)
        non_value_missing = [k for k in missing if not k.startswith("value_head.")]
        if non_value_missing or unexpected:
            raise RuntimeError(f"Error loading {path}: missing {non_value_missing}, unexpected {unexpected}")
        return net.to(device).eval()
