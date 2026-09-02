"""Capa 1: behavioral cloning + self-imitation (Documento 12).

BC: NLL de acciones de un maestro (ScriptedTeacher) bajo π actual.
SIL: mismo NLL sobre transiciones élite propias (solo win).
Even-pick por episodio, no la cola del ring; prefiere wins <40k ticks.

No sustituye a PPO: L = L_PPO + λ_bc L_BC + λ_sil L_SIL.
λ_bc arranca en 1.0 y baja a 0 en --bc-warmup iters (kickstarting).
El origen del warmup (bc_start_iter) se persiste en el ckpt: un resume no
vuelve a λ=1.0.
"""
from __future__ import annotations

import torch

from rl.action_adapter import ENABLED_TYPES, TYPE_TO_IDX
from rl.network import ACTION_TYPES
from rl.roles import role_of
from openra_env.models import ActionType, CommandModel

# Eco / producción primero: si el teacher emite TRAIN y ARMY en el mismo
# tick, clonar TRAIN. El asalto ya lo sostiene auto_support.
_BC_PRIORITY = (
    "train", "build", "place_building", "harvest", "deploy",
)
_BC_LAST = {"army_attack_move", "attack_move", "attack", "no_op"}
_BC_COMBAT_CAP_TYPES = frozenset(_BC_LAST)


def lambda_bc_at(it: int, start_iter: int, warmup: int = 80,
                 start: float = 1.0, end: float = 0.0) -> float:
    """Linear decay from `start` to `end` over `warmup` iters after start_iter."""
    if warmup <= 0:
        return float(end)
    t = (int(it) - int(start_iter)) / float(warmup)
    t = max(0.0, min(1.0, t))
    return float(start + (end - start) * t)


def _cmd_name(c) -> str:
    return getattr(getattr(c, "action", None), "value", None) or str(
        getattr(c, "action", ""))


def pick_bc_command(commands) -> CommandModel:
    """Orden clonable: TRAIN/BUILD/PLACE ganan a army_attack_move/no_op."""
    enabled = []
    for c in commands or []:
        name = _cmd_name(c)
        if name in ENABLED_TYPES:
            enabled.append((name, c))
    if not enabled:
        return CommandModel(action=ActionType.NO_OP)
    for pref in _BC_PRIORITY:
        for name, c in enabled:
            if name == pref:
                return c
    for name, c in enabled:
        if name not in _BC_LAST:
            return c
    return enabled[0][1]


def sample_type_name(s: dict) -> str:
    act = s.get("action") or {}
    t = act.get("type")
    if torch.is_tensor(t):
        t = int(t.reshape(-1)[0].item())
    try:
        t = int(t)
    except (TypeError, ValueError):
        t = 0
    if 0 <= t < len(ACTION_TYPES):
        return ACTION_TYPES[t]
    return "no_op"


def _even_pick(group: list, cap: int) -> list:
    if cap <= 0 or not group:
        return []
    if len(group) <= cap:
        return list(group)
    if cap == 1:
        return [group[0]]
    return [group[round(i * (len(group) - 1) / (cap - 1))] for i in range(cap)]


def balance_bc_samples(samples: list, per_type_cap: int = 96,
                       combat_cap: int = 64) -> list:
    """Cap combat/no_op so a 600-step incomplete attack tape cannot drown TRAIN."""
    if not samples:
        return []
    buckets: dict[str, list] = {}
    for s in samples:
        buckets.setdefault(sample_type_name(s), []).append(s)
    picked = []
    for name, group in buckets.items():
        cap = combat_cap if name in _BC_COMBAT_CAP_TYPES else per_type_cap
        picked.extend(_even_pick(group, cap))
    order = {id(s): i for i, s in enumerate(samples)}
    picked.sort(key=lambda s: order.get(id(s), 0))
    return picked


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


# 1141 closed in 17–30k. A 50k win dumps ~1k late train-spam into the ring;
# sample_recent(512) used to clone that tail (Run 33 plateau).
SIL_PREFER_TICKS = 40000


class EliteBuffer:
    """Winning episodes for SIL, even-pick per win (not the tail of 1–2 longs)."""

    def __init__(self, cap_steps: int = 2000,
                 prefer_ticks: int = SIL_PREFER_TICKS):
        self.cap = int(cap_steps)
        self.prefer_ticks = int(prefer_ticks)
        self._episodes: list[dict] = []

    def __len__(self) -> int:
        return self._n_steps()

    def _n_steps(self) -> int:
        return sum(len(e.get("steps") or ()) for e in self._episodes)

    def add_episode(self, samples: list, outcome: dict | None) -> int:
        if not samples:
            return 0
        oc = outcome or {}
        result = str(oc.get("result", "") or "")
        # Lose+raze is almost every a_short game. Cloning it (Run 32) filled
        # the ring with "poke buildings and die" and SIL pulled latest off
        # the 1081 peak. Wins only — including win_early.
        if not result.startswith("win"):
            return 0
        try:
            ticks = int(oc.get("ticks") or 0)
        except (TypeError, ValueError):
            ticks = 0
        cloned = [_cpu_clone_step(s) for s in samples]
        self._episodes.append({"steps": cloned, "ticks": ticks})
        self._trim()
        return len(cloned)

    def _trim(self) -> None:
        """Drop oldest long wins first; if one ep exceeds cap, even-pick it."""
        while self._episodes and self._n_steps() > self.cap:
            if len(self._episodes) == 1:
                ep = self._episodes[0]
                ep["steps"] = _even_pick(ep["steps"], self.cap)
                break
            long_i = next(
                (i for i, e in enumerate(self._episodes)
                 if int(e.get("ticks") or 0) >= self.prefer_ticks),
                None,
            )
            if long_i is not None:
                self._episodes.pop(long_i)
            else:
                self._episodes.pop(0)

    def snapshot(self) -> list:
        out = []
        for e in self._episodes:
            out.extend(e.get("steps") or ())
        return out

    def sample_recent(self, max_steps: int = 512) -> list:
        """Even-pick across winning episodes. Prefer ticks < prefer_ticks.

        A 50k win used to fill the last 512 with train-spam. Short wins
        (1141: 17–30k) get the quota; long wins are fallback if none short.
        """
        if max_steps <= 0 or not self._episodes:
            return []
        eps = [e for e in self._episodes if e.get("steps")]
        if not eps:
            return []
        short = [
            e for e in eps
            if 0 < int(e.get("ticks") or 0) < self.prefer_ticks
        ]
        pool = short if short else eps
        n = len(pool)
        if n <= 0:
            return []
        base = int(max_steps) // n
        extra = int(max_steps) % n
        out = []
        for i, e in enumerate(pool):
            q = base + (1 if i < extra else 0)
            out.extend(_even_pick(e["steps"], q))
        return out
