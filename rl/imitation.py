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

import json
import os
from pathlib import Path

import torch

from rl.action_adapter import ENABLED_TYPES, TYPE_TO_IDX, n_combat_total
from rl.network import ACTION_TYPES
from rl.roles import role_of
from openra_env.models import ActionType, CommandModel

# Eco / producción primero en el opening. En attack (leftover o n_combat>=8)
# pick_bc_commands clona TAMBIÉN un push: sin eso SFT es miller (TRAIN) y el
# asalto no lo sostiene auto_support (A lleva --no-war-nudge).
_BC_PRIORITY = (
    "train", "build", "place_building", "harvest", "deploy",
)
_BC_LAST = {"army_attack_move", "attack_move", "attack", "no_op"}
_BC_COMBAT_CAP_TYPES = frozenset(_BC_LAST)
# Alineado a ScriptedTeacher.RUSH_ATTACK_MOVE. Opening (<8, sin leftover)
# no clona combate: sin eco sana no hay army que empujar.
_BC_COMBAT_READY_N = 8
# Manifest de TeacherWinBuffer. Cintas viejas (solo TRAIN) no se hidratan.
TAPE_SCHEMA = "eco_and_combat_mental_v4"


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
    """Opening: TRAIN/BUILD/PLACE ganan a army_attack_move/no_op.

    El push en attack va por pick_bc_commands (segunda label), no acá.
    """
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


# Chebyshev radius: combat BC label must sit near visible/ghost, not beacon.
_BC_CONTACT_NEAR = 16


def _cmd_xy(c):
    try:
        tx = getattr(c, "target_x", None)
        ty = getattr(c, "target_y", None)
        if tx is None or ty is None:
            return None
        return int(tx), int(ty)
    except (TypeError, ValueError):
        return None


def _obs_contact_cells(obs, belief=None) -> list:
    """Visible leftovers + optional belief ghosts (last_seen)."""
    cells = []
    if obs is None:
        return cells
    for b in list(getattr(obs, "visible_enemy_buildings", None) or []):
        try:
            cells.append((int(b.cell_x), int(b.cell_y)))
        except (TypeError, ValueError, AttributeError):
            continue
    for u in list(getattr(obs, "visible_enemies", None) or []):
        try:
            cells.append((int(u.cell_x), int(u.cell_y)))
        except (TypeError, ValueError, AttributeError):
            continue
    if belief is not None:
        try:
            slots = belief.select_enemies(obs)
        except Exception:
            slots = []
        for u, vis, conf, _t in slots:
            if float(vis) >= 0.5 or float(conf) < 0.05:
                continue
            try:
                cells.append((int(u.cell_x), int(u.cell_y)))
            except (TypeError, ValueError, AttributeError):
                continue
    return cells


def _near_any(xy, cells, radius: int = _BC_CONTACT_NEAR) -> bool:
    if xy is None or not cells:
        return False
    r = int(radius)
    x, y = int(xy[0]), int(xy[1])
    return any(max(abs(x - cx), abs(y - cy)) <= r for cx, cy in cells)


def _retarget_combat(cmd, cells):
    """Clone combat cmd aimed at first contact cell (reject beacon mill)."""
    if cmd is None or not cells:
        return cmd
    tx, ty = int(cells[0][0]), int(cells[0][1])
    return CommandModel(
        action=getattr(cmd, "action", ActionType.NO_OP),
        actor_id=getattr(cmd, "actor_id", None),
        target_x=tx,
        target_y=ty,
        target_actor_id=getattr(cmd, "target_actor_id", None),
        item_type=getattr(cmd, "item_type", None),
    )


def _combat_bc_command(commands, obs=None, belief=None):
    """Un push clonable: army_attack_move > attack_move > attack.

    Con leftover/ghosts visibles: preferí el cmd cuyo target está cerca;
    si el teacher aún apunta al beacon, retarget al contacto.
    """
    enabled = []
    for c in commands or []:
        name = _cmd_name(c)
        if name in ENABLED_TYPES:
            enabled.append((name, c))
    contacts = _obs_contact_cells(obs, belief)
    for want in ("army_attack_move", "attack_move", "attack"):
        cands = [c for name, c in enabled if name == want]
        if not cands:
            continue
        if contacts:
            near = [c for c in cands if _near_any(_cmd_xy(c), contacts)]
            if near:
                return near[0]
            return _retarget_combat(cands[0], contacts)
        return cands[0]
    return None


def _bc_combat_ready(obs) -> bool:
    if obs is None:
        return False
    leftover = bool(
        getattr(obs, "visible_enemy_buildings", None)
        or getattr(obs, "visible_enemies", None)
    )
    try:
        n = int(n_combat_total(obs))
    except (TypeError, ValueError):
        n = 0
    return leftover or n >= _BC_COMBAT_READY_N



def student_combat_ready(obs, belief=None) -> bool:
    """When True, student dual-emits eco + combat push same macro-tick.

    Ready if BC combat gate fires, or belief has mental enemy_base / leftover
    ghosts (last contact). Opening without army stays single-action.
    """
    if _bc_combat_ready(obs):
        return True
    if belief is None:
        return False
    if getattr(belief, "enemy_base_xy", None) is not None:
        return True
    # Leftover ghosts still in memory count as last contact.
    mem = getattr(belief, "_mem", None) or {}
    if mem:
        return True
    return False


def pick_bc_commands(commands, obs=None, belief=None) -> list:
    """1–2 labels por tick. Opening = eco. Attack = eco + un push.

    Sin leftover y n_combat<8: igual que pick_bc_command (TRAIN gana).
    Con leftover o pack de rush: si hay combate distinto del primary, se
    clonan los dos. Push BC prefiere celdas cerca de visibles/ghosts, no beacon.
    """
    primary = pick_bc_command(commands)
    combat = _combat_bc_command(commands, obs=obs, belief=belief)
    if combat is None or combat is primary:
        return [primary]
    if not _bc_combat_ready(obs):
        return [primary]
    if _cmd_name(primary) in ("army_attack_move", "attack_move", "attack"):
        # Primary is already combat: still retarget away from beacon if needed.
        contacts = _obs_contact_cells(obs, belief)
        if contacts and not _near_any(_cmd_xy(primary), contacts):
            return [_retarget_combat(primary, contacts)]
        return [primary]
    return [primary, combat]


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


def merge_teacher_wins(episodes: list, keep_incomplete: bool = False,
                       incomplete_min_ticks: int = 15000) -> tuple[list, dict, list]:
    """Keep teacher tapes worth cloning.

    Always: result startswith win.
    Never: lose / engine_error (clonar palizas enseña a morir).
    Optional (`keep_incomplete`, legacy): incomplete largo — el build order
    sigue ahí aunque a_short no declare win.

    Returns (flat_steps, meta, kept_episodes) where kept_episodes is a list of
    {"steps", "ticks", "result"} for per-episode buffers (TeacherWinBuffer).
    """
    kept: list = []
    kept_eps: list = []
    results: list[str] = []
    n_raw = 0
    n_win = 0
    n_inc = 0
    for samples, oc in episodes or []:
        chunk = list(samples or [])
        n_raw += len(chunk)
        oc = oc or {}
        r = str(oc.get("result") or "")
        results.append(r)
        try:
            ticks = int(oc.get("ticks") or 0)
        except (TypeError, ValueError):
            ticks = 0
        take = r.startswith("win")
        if (not take and keep_incomplete and r == "incomplete"
                and ticks >= int(incomplete_min_ticks)):
            take = True
            n_inc += 1
        if take:
            kept.extend(chunk)
            kept_eps.append({"steps": chunk, "ticks": ticks, "result": r})
            if r.startswith("win"):
                n_win += 1
    return kept, {
        "bc_n_raw": n_raw,
        "bc_n": len(kept),
        "bc_n_eps": len(episodes or []),
        "bc_n_win_eps": n_win,
        "bc_n_incomplete_eps": n_inc,
        "bc_results": results,
        "bc_result": results[-1] if results else "",
    }, kept_eps


def balance_bc_samples(samples: list, per_type_cap: int = 512,
                       combat_cap: int = 512) -> list:
    """Cap por tipo. Combate y TRAIN al mismo techo: sin eco no hay army,
    sin combate el SFT es miller. Phase A needs ~512 (was 96) so attack BC
    is not starved after QSA-dense + beacon opening. Even-pick keeps mix."""
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
# Teacher tapes for reuse: Phase A "Bien" ~35–50 short rushes; with eco+push K=2
# short wins ~1.0–1.5k steps → 64000 ≈ 40–60 before trim.
# Cap is STEPS not episodes. Long = ticks >= prefer; trim drops those first.
# 20k corta el timeout-adjacent.
BC_WIN_CAP = 64000
BC_WIN_PREFER_TICKS = 20000


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

class TeacherWinBuffer:
    """Persistent ring of teacher win episodes for BC/SFT (across iters).

    Same trim/sample pattern as EliteBuffer (prefer short wins), plus
    `{ckpt_dir}/teacher_wins/` manifest + ep_XXXX.pt so a resume still has
    past wins when the current iter collects 0.
    """

    def __init__(self, cap_steps: int = BC_WIN_CAP,
                 prefer_ticks: int = BC_WIN_PREFER_TICKS,
                 path: str | os.PathLike | None = None,
                 keep_incomplete: bool = False,
                 incomplete_min_ticks: int = 15000):
        self.cap = int(cap_steps)
        self.prefer_ticks = int(prefer_ticks)
        self.path = Path(path) if path else None
        self.keep_incomplete = bool(keep_incomplete)
        self.incomplete_min_ticks = int(incomplete_min_ticks)
        self._episodes: list[dict] = []
        self._next_id = 0
        if self.path is not None and self.path.is_dir():
            self.load()

    def __len__(self) -> int:
        return self._n_steps()

    @property
    def n_episodes(self) -> int:
        return len(self._episodes)

    def _n_steps(self) -> int:
        return sum(len(e.get("steps") or ()) for e in self._episodes)

    def _accept(self, result: str, ticks: int) -> bool:
        if result.startswith("win"):
            return True
        if (self.keep_incomplete and result == "incomplete"
                and ticks >= self.incomplete_min_ticks):
            return True
        return False

    def add_episode(self, samples: list, outcome: dict | None) -> int:
        if not samples:
            return 0
        oc = outcome or {}
        result = str(oc.get("result", "") or "")
        try:
            ticks = int(oc.get("ticks") or 0)
        except (TypeError, ValueError):
            ticks = 0
        if not self._accept(result, ticks):
            return 0
        cloned = [_cpu_clone_step(s) for s in samples]
        self._episodes.append({
            "id": int(self._next_id),
            "steps": cloned,
            "ticks": ticks,
            "result": result,
        })
        self._next_id += 1
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

    def sample(self, max_steps: int = 512) -> list:
        """Even-pick across kept episodes. Prefer ticks < prefer_ticks."""
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

    def save(self, path: str | os.PathLike | None = None) -> None:
        root = Path(path) if path is not None else self.path
        if root is None:
            return
        root.mkdir(parents=True, exist_ok=True)
        for old in root.glob("ep_*.pt"):
            try:
                old.unlink()
            except OSError:
                pass
        manifest = {
            "schema": TAPE_SCHEMA,
            "cap": self.cap,
            "prefer_ticks": self.prefer_ticks,
            "episodes": [],
        }
        for e in self._episodes:
            eid = int(e.get("id", 0))
            fname = f"ep_{eid:04d}.pt"
            torch.save({
                "steps": e.get("steps") or [],
                "ticks": int(e.get("ticks") or 0),
                "result": str(e.get("result") or "win"),
            }, root / fname)
            manifest["episodes"].append({
                "id": eid,
                "file": fname,
                "ticks": int(e.get("ticks") or 0),
                "result": str(e.get("result") or "win"),
                "n_steps": len(e.get("steps") or ()),
            })
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        self.path = root

    def load(self, path: str | os.PathLike | None = None) -> int:
        root = Path(path) if path is not None else self.path
        if root is None or not root.is_dir():
            return 0
        man_path = root / "manifest.json"
        if not man_path.is_file():
            return 0
        try:
            manifest = json.loads(man_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 0
        schema = str(manifest.get("schema") or "")
        if schema != TAPE_SCHEMA:
            # Schema mismatch (beacon tapes / solo TRAIN) no se hidrata: un
            # --scratch las reusaría como miller. Recolectar de nuevo.
            return 0
        loaded: list[dict] = []
        max_id = -1
        for entry in manifest.get("episodes") or []:
            fname = str(entry.get("file") or "")
            if not fname:
                eid = int(entry.get("id") or 0)
                fname = f"ep_{eid:04d}.pt"
            fpath = root / fname
            if not fpath.is_file():
                continue
            try:
                blob = torch.load(fpath, map_location="cpu", weights_only=False)
            except TypeError:
                blob = torch.load(fpath, map_location="cpu")
            except Exception:
                continue
            steps = list(blob.get("steps") or [])
            if not steps:
                continue
            eid = int(entry.get("id", blob.get("id", 0)) or 0)
            max_id = max(max_id, eid)
            loaded.append({
                "id": eid,
                "steps": steps,
                "ticks": int(blob.get("ticks") or entry.get("ticks") or 0),
                "result": str(blob.get("result") or entry.get("result") or "win"),
            })
        self._episodes = loaded
        self._next_id = max_id + 1
        self.path = root
        self._trim()
        return len(self._episodes)
