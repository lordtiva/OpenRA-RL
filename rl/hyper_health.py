"""Multi-signal PPO hyperparameter self-regulation (NOT LR-only).

Post-update / end-of-iter: look at PPO KL, clip_frac, entropy, grad_norm
(and v_loss if present) and take ONE clamped action with cooldown.

Priority: KL → grad_norm → entropy → clip_frac (tie-break).
Prefer KL bands (~0.01–0.03 high, ~0.008 low) over clip_frac>0.3 for LR
(clip is often inflated while KL stays healthy).
grad_norm lr_down needs consecutive iters (default 2); lr_min is 5e-6.

Does NOT touch γ, λ GAE, or SIL/BC λ. Watches PPO KL, never BC NLL.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ACTIONS = ("lr_down", "lr_up", "ent_up", "early_stop", "noop")


@dataclass
class HyperConfig:
    """Thresholds + clamps. Tuned for onboard B+ AlphaLite PPO."""

    # KL bands (PPO approx KL from trainer.update — not BC NLL).
    kl_high: float = 0.03
    kl_low: float = 0.008
    kl_soft: float = 0.01          # soft-high; enables clip tie-break
    kl_early_stop: float = 0.05    # very high → early_stop if epochs>1

    grad_high: float = 5.0
    grad_high_consecutive: int = 2  # isolated timeout spike must not lr_down
    # gn >> max_grad_norm (clip 1.0): 1920 with clip 0.88 is broken PPO.
    # Immediate lr_down / early_stop, no consecutive-iter streak.
    grad_explode: float = 50.0
    entropy_low: float = 1.0
    clip_high: float = 0.3

    lr_min: float = 5e-6  # 1e-6 froze recovery after timeout grad spikes
    lr_max: float = 3e-4
    lr_down_factor: float = 0.7
    lr_up_factor: float = 1.2

    # trainer.ent_lo is the controllable entropy floor (logged as entropy_coef)
    ent_lo_min: float = 0.01
    ent_lo_max: float = 0.20
    ent_up_delta: float = 0.01

    cooldown_iters: int = 3
    pause_iters_default: int = 8   # 5–10 after promote / --reset-opt


@dataclass
class HyperDecision:
    action: str = "noop"
    reason: str = ""
    lr: float | None = None
    entropy_coef: float | None = None
    paused: bool = False
    next_epochs: int | None = None

    def as_metrics(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "hyper_action": self.action,
            "hyper_reason": self.reason,
            "hyper_paused": bool(self.paused),
        }
        if self.lr is not None:
            out["lr"] = round(float(self.lr), 8)
        if self.entropy_coef is not None:
            out["entropy_coef"] = round(float(self.entropy_coef), 5)
        return out


def resolve_auto_hyper(args) -> bool:
    """Default ON for onboard B+; OFF for phase A / non-onboard.

    Explicit --auto-hyper / --no-auto-hyper wins when not None.
    """
    flag = getattr(args, "auto_hyper", None)
    if flag is not None:
        return bool(flag)
    phase = getattr(args, "onboard_phase", None)
    if phase is None:
        return False
    return str(phase) not in ("A", "")


class HyperHealthMonitor:
    """Stateful one-action-per-iter regulator."""

    def __init__(self, cfg: HyperConfig | None = None, enabled: bool = True):
        self.cfg = cfg or HyperConfig()
        self.enabled = bool(enabled)
        self._cooldown = 0
        self._pause_left = 0
        self._epochs_override: int | None = None
        self._grad_high_count = 0
        self.last = HyperDecision(action="noop", reason="init")

    def pause(self, n: int | None = None) -> None:
        """Hold actions for n iters (promote / --reset-opt)."""
        want = int(self.cfg.pause_iters_default if n is None else n)
        self._pause_left = max(self._pause_left, max(0, want))

    def consume_epochs_override(self, default_epochs: int) -> int:
        """If prior early_stop fired, shrink next update's epoch count."""
        if self._epochs_override is None:
            return int(default_epochs)
        e = int(self._epochs_override)
        self._epochs_override = None
        return max(1, min(e, int(default_epochs)))

    def current_lr(self, trainer) -> float:
        try:
            return float(trainer.opt.param_groups[0]["lr"])
        except Exception:
            return 0.0

    def current_ent_lo(self, trainer) -> float:
        return float(getattr(trainer, "ent_lo", self.cfg.ent_lo_min))

    def decide(
        self,
        *,
        kl: float | None,
        clip_frac: float | None,
        entropy: float | None,
        grad_norm: float | None,
        v_loss: float | None = None,
        lr: float,
        ent_lo: float,
        epochs: int = 1,
        skipped: bool = False,
        grad_high_streak: int = 0,
    ) -> HyperDecision:
        """Pure decision (no mutation). ONE action or noop."""
        cfg = self.cfg
        lr = float(lr)
        ent_lo = float(ent_lo)
        base = HyperDecision(lr=lr, entropy_coef=ent_lo)

        if not self.enabled:
            base.action = "noop"
            base.reason = "disabled"
            return base

        if skipped:
            base.action = "noop"
            base.reason = "update_skipped"
            return base

        if self._pause_left > 0:
            base.action = "noop"
            base.reason = f"paused({self._pause_left})"
            base.paused = True
            return base

        if self._cooldown > 0:
            base.action = "noop"
            base.reason = f"cooldown({self._cooldown})"
            return base

        kl_v = _f(kl)
        clip_v = _f(clip_frac)
        ent_v = _f(entropy)
        gn_v = _f(grad_norm)
        # v_loss reserved for future tie-break; unused for actions today
        _ = v_loss

        # --- Priority 0: exploded grad (skip streak; clip is meaningless) ---
        if gn_v is not None and gn_v >= cfg.grad_explode:
            if lr <= cfg.lr_min * 1.01:
                base.action = "early_stop"
                base.reason = (
                    f"grad_explode={gn_v:.1f}>={cfg.grad_explode} lr floor")
                base.next_epochs = 1
                return base
            base.action = "lr_down"
            base.reason = f"grad_explode={gn_v:.1f}>={cfg.grad_explode}"
            return base

        # --- Priority 1: KL ---
        if kl_v is not None:
            if kl_v >= cfg.kl_early_stop and int(epochs) > 1:
                base.action = "early_stop"
                base.reason = f"kl={kl_v:.4f}>={cfg.kl_early_stop} epochs>1"
                base.next_epochs = 1
                return base
            if kl_v >= cfg.kl_high:
                if lr <= cfg.lr_min * 1.01:
                    base.action = "noop"
                    base.reason = f"kl={kl_v:.4f}>={cfg.kl_high} but lr at floor"
                    return base
                base.action = "lr_down"
                base.reason = f"kl={kl_v:.4f}>={cfg.kl_high}"
                return base
            if kl_v <= cfg.kl_low:
                if lr >= cfg.lr_max * 0.99:
                    base.action = "noop"
                    base.reason = f"kl={kl_v:.4f}<={cfg.kl_low} but lr at ceil"
                    return base
                base.action = "lr_up"
                base.reason = f"kl={kl_v:.4f}<={cfg.kl_low}"
                return base

        # --- Priority 2: grad_norm (consecutive iters, not one timeout spike) ---
        if gn_v is not None and gn_v >= cfg.grad_high:
            need = max(1, int(cfg.grad_high_consecutive))
            streak = int(grad_high_streak or 0)
            if streak < need:
                base.action = "noop"
                base.reason = (
                    f"grad_norm={gn_v:.2f}>={cfg.grad_high} "
                    f"streak={streak}/{need}")
                return base
            if lr <= cfg.lr_min * 1.01:
                base.action = "noop"
                base.reason = f"grad_norm={gn_v:.2f}>={cfg.grad_high} lr floor"
                return base
            base.action = "lr_down"
            base.reason = f"grad_norm={gn_v:.2f}>={cfg.grad_high}"
            return base

        # --- Priority 3: entropy ---
        if ent_v is not None and ent_v < cfg.entropy_low:
            if ent_lo >= cfg.ent_lo_max * 0.99:
                base.action = "noop"
                base.reason = f"entropy={ent_v:.3f}<{cfg.entropy_low} ent_lo ceil"
                return base
            base.action = "ent_up"
            base.reason = f"entropy={ent_v:.3f}<{cfg.entropy_low}"
            return base

        # --- Priority 4: clip_frac tie-break (only if KL soft-high) ---
        if (clip_v is not None and clip_v > cfg.clip_high
                and kl_v is not None and kl_v >= cfg.kl_soft):
            if lr <= cfg.lr_min * 1.01:
                base.action = "noop"
                base.reason = (
                    f"clip={clip_v:.3f}>{cfg.clip_high}+kl_soft "
                    f"but lr floor")
                return base
            base.action = "lr_down"
            base.reason = (
                f"clip={clip_v:.3f}>{cfg.clip_high} "
                f"kl={kl_v:.4f}>={cfg.kl_soft} (tie-break)")
            return base

        # Healthy / clip alone with healthy KL → noop (do NOT chase clip)
        if clip_v is not None and clip_v > cfg.clip_high and (
                kl_v is None or kl_v < cfg.kl_soft):
            base.action = "noop"
            base.reason = (
                f"clip={clip_v:.3f}>{cfg.clip_high} but KL healthy "
                f"(kl={kl_v})")
            return base

        base.action = "noop"
        base.reason = "healthy"
        return base

    def apply(self, trainer, decision: HyperDecision) -> HyperDecision:
        """Mutate trainer lr / ent_lo; tick pause+cooldown; stash early_stop."""
        cfg = self.cfg
        lr = self.current_lr(trainer)
        ent_lo = self.current_ent_lo(trainer)
        decision.lr = lr
        decision.entropy_coef = ent_lo

        if self._pause_left > 0:
            self._pause_left -= 1
            decision.paused = True
            # still count as paused even if decide already said so
            if decision.action == "noop" and not decision.reason.startswith("paused"):
                decision.reason = f"paused({self._pause_left + 1})"
            self.last = decision
            return decision

        if decision.action == "noop":
            if self._cooldown > 0:
                self._cooldown -= 1
            self.last = decision
            return decision

        if decision.action == "lr_down":
            new_lr = max(cfg.lr_min, lr * cfg.lr_down_factor)
            _set_lr(trainer, new_lr)
            decision.lr = new_lr
            self._cooldown = cfg.cooldown_iters
        elif decision.action == "lr_up":
            new_lr = min(cfg.lr_max, lr * cfg.lr_up_factor)
            _set_lr(trainer, new_lr)
            decision.lr = new_lr
            self._cooldown = cfg.cooldown_iters
        elif decision.action == "ent_up":
            new_e = min(cfg.ent_lo_max, ent_lo + cfg.ent_up_delta)
            trainer.ent_lo = new_e
            decision.entropy_coef = new_e
            self._cooldown = cfg.cooldown_iters
        elif decision.action == "early_stop":
            self._epochs_override = int(decision.next_epochs or 1)
            self._cooldown = cfg.cooldown_iters

        self.last = decision
        return decision

    def step(
        self,
        trainer,
        stats: dict,
        *,
        epochs: int = 1,
        skipped: bool = False,
    ) -> HyperDecision:
        """decide + apply from a PPO stats dict."""
        gn_v = _f(stats.get("grad_norm"))
        if skipped or self._pause_left > 0 or self._cooldown > 0:
            pass
        elif gn_v is not None and gn_v >= self.cfg.grad_high:
            self._grad_high_count += 1
        else:
            self._grad_high_count = 0
        decision = self.decide(
            kl=stats.get("kl"),
            clip_frac=stats.get("clip_frac"),
            entropy=stats.get("entropy"),
            grad_norm=stats.get("grad_norm"),
            v_loss=stats.get("v_loss"),
            lr=self.current_lr(trainer),
            ent_lo=self.current_ent_lo(trainer),
            epochs=epochs,
            skipped=skipped,
            grad_high_streak=self._grad_high_count,
        )
        return self.apply(trainer, decision)


def _f(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return v


def _set_lr(trainer, lr: float) -> None:
    for g in trainer.opt.param_groups:
        g["lr"] = float(lr)
