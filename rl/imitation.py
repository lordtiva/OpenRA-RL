"""Capa 1: behavioral cloning + self-imitation (Documento 12).

BC: NLL de acciones de un maestro (ScriptedTeacher) bajo π actual.
SIL: mismo NLL sobre transiciones élite propias (win o raze>0).

No sustituye a PPO: L = L_PPO + λ_bc L_BC + λ_sil L_SIL.
λ_bc arranca en 1.0 y baja a 0 en --bc-warmup iters (kickstarting).
"""
from __future__ import annotations

from collections import deque

import torch

from rl.action_adapter import ENABLED_TYPES, TYPE_TO_IDX
from rl.roles import role_of
from openra_env.models import ActionType, CommandModel


def lambda_bc_at(it: int, start_iter: int, warmup: int = 80,
                 start: float = 1.0, end: float = 0.0) -> float:
    """Linear decay from `start` to `end` over `warmup` iters after start_iter."""
    if warmup <= 0:
        return float(end)
    t = (int(it) - int(start_iter)) / float(warmup)
    t = max(0.0, min(1.0, t))
    return float(start + (end - start) * t)


def pick_bc_command(commands) -> CommandModel:
    """Primera orden del maestro que la red v0.1 puede representar."""
    for c in commands or []:
        name = getattr(getattr(c, "action", None), "value", None) or str(
            getattr(c, "action", ""))
        if name in ENABLED_TYPES:
            return c
    return CommandModel(action=ActionType.NO_OP)


def command_to_indices(obs, cmd: CommandModel, aidx) -> tuple[int, int, int, int]:
    """CommandModel -> (type, unit_slot, cell_flat, item_slot) en el ActionIndex."""
    name = getattr(getattr(cmd, "action", None), "value", None) or str(
        getattr(cmd, "action", "no_op"))
    if name not in ENABLED_TYPES:
        name = "no_op"
    t_idx = int(TYPE_TO_IDX.get(name, 0))

    unit_slot = 0
    actor_id = int(getattr(cmd, "actor_id", 0) or 0)
    if actor_id and actor_id in aidx.unit_ids:
        unit_slot = int(aidx.unit_ids.index(actor_id))

    cx = int(getattr(cmd, "target_x", 0) or 0)
    cy = int(getattr(cmd, "target_y", 0) or 0)
    cx = max(0, min(aidx.w - 1, cx))
    cy = max(0, min(aidx.h - 1, cy))
    cell_flat = int(cy) * int(aidx.w) + int(cx)

    item_slot = 0
    item = str(getattr(cmd, "item_type", "") or "")
    if item and aidx.items:
        role = role_of(item)
        if role in aidx.items:
            item_slot = int(aidx.items.index(role))
        elif item in aidx.items:
            item_slot = int(aidx.items.index(item))
    return t_idx, unit_slot, cell_flat, item_slot


def _cpu_clone_step(s: dict) -> dict:
    """Detach rollout tensors so the SIL ring does not pin live GPU storage."""
    out = dict(s)
    batch = s.get("batch") or {}
    out["batch"] = {
        k: (v.detach().cpu().contiguous() if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }
    act = s.get("action") or {}
    out["action"] = {
        k: (v.detach().cpu().contiguous() if torch.is_tensor(v) else v)
        for k, v in act.items()
    }
    h = s.get("h_in")
    if torch.is_tensor(h):
        out["h_in"] = h.detach().cpu().contiguous()
    return out


class EliteBuffer:
    """Ring of on-policy steps from winning / razing episodes (SIL)."""

    def __init__(self, cap_steps: int = 2000):
        self.cap = int(cap_steps)
        self._steps: deque = deque()

    def __len__(self) -> int:
        return len(self._steps)

    def add_episode(self, samples: list, outcome: dict | None) -> int:
        if not samples:
            return 0
        oc = outcome or {}
        result = str(oc.get("result", "") or "")
        rc = oc.get("reward_components") or {}
        raze = float(rc.get("raze", 0) or 0)
        # raze>0 is almost every a_short game (beginner drops a building).
        # Only keep real wins or a clear raze so the ring does not ingest
        # 4x~500 maps every iter (that was the VRAM walk).
        if not (result.startswith("win") or raze >= 2.0):
            return 0
        n = 0
        for s in samples:
            self._steps.append(_cpu_clone_step(s))
            n += 1
            while len(self._steps) > self.cap:
                self._steps.popleft()
        return n

    def snapshot(self) -> list:
        return list(self._steps)

    def sample_recent(self, max_steps: int = 512) -> list:
        """Most recent elite steps, capped so SIL does not replay 2k maps/iter."""
        if max_steps <= 0 or not self._steps:
            return []
        if len(self._steps) <= max_steps:
            return list(self._steps)
        return list(self._steps)[-int(max_steps):]
