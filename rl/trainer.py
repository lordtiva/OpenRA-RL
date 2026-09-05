"""PPO trainer para el agente OpenRA.

Diseño (lecciones alto-truco/Imperium aplicadas):
    - old_log_prob y old_value se GUARDAN en el rollout y se congelan:
      recalcularlos con pesos actuales haría ratio≡1 (PPO sin señal)
    - Ventajas: el escalado depende de --adv-mode en train.py
      ('episode' centra por episodio; 'global' z-scorea el batch completo;
      'none' no toca nada). F8 (auditoría 2026-08-24): ESTA clase ya NO
      re-normaliza — antes z-scoreaba siempre y con adv-mode=global la
      normalización era doble (el comentario "no doble normalización" era
      falso). El escalado vive en UN solo lugar: process_results.
    - Clip ε=0.2, grad clip 0.5, value clipping + Huber
    - Coeficiente de entropía dinámico calculado on-device (sin .item()
      dentro del grafo)
"""

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

# Dest-credit puede guardar lp_old ~ -1e9 (celda tapada) y lp_new finito.
# ratio=exp(Δ) → inf; con adv<0 PPO no clippea y pi_loss=inf (iter 923 Run 17).
_LOG_RATIO_CLAMP = 8.0
# SIL lp.clamp(-20) deja nll≈20 cuando la acción sigue ilegal. No clonar eso.
_SIL_NLL_SKIP = 18.0


def _make_scaler(device: str):
    enabled = device == "cuda" and torch.cuda.is_available()
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (TypeError, AttributeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(device: str):
    enabled = device == "cuda" and torch.cuda.is_available()
    try:
        return torch.amp.autocast("cuda", enabled=enabled)
    except (TypeError, AttributeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def _to_device(t, device: str):
    if not torch.is_tensor(t):
        return t
    if device == "cuda" and t.device.type == "cpu":
        if not t.is_pinned():
            try:
                t = t.pin_memory()
            except RuntimeError:
                pass
        return t.to(device, non_blocking=True)
    if t.device.type != device:
        return t.to(device, non_blocking=True)
    return t


def prefetch_steps(steps: list, device: str, inplace: bool = False) -> list:
    """Sube tensores a GPU UNA vez. inplace=True para el rollout (se descarta).

    SIL/elite: inplace=False — el ring se queda en CPU (no pinear VRAM).
    """
    out = []
    for s in steps:
        ns = s if inplace else dict(s)
        b = s.get("batch") or {}
        nb = b if inplace else dict(b)
        for k, v in list(b.items()):
            if torch.is_tensor(v):
                nb[k] = _to_device(v, device)
        ns["batch"] = nb
        a = s.get("action") or {}
        na = a if inplace else dict(a)
        for k, v in list(a.items()):
            if torch.is_tensor(v):
                na[k] = _to_device(v, device)
        ns["action"] = na
        if torch.is_tensor(s.get("h_in")):
            ns["h_in"] = _to_device(s["h_in"], device)
        for k in ("adv", "ret", "value_pred"):
            v = s.get(k)
            if v is None:
                continue
            if torch.is_tensor(v):
                ns[k] = _to_device(v, device)
            else:
                ns[k] = torch.tensor(float(v), device=device, dtype=torch.float32)
        if not inplace:
            ns["_ep"] = s.get("_ep")
        out.append(ns)
    return steps if inplace else out


def _split_segments(samples: list, bptt_len: int,
                   burn_in_len: int = 0) -> list:
    """Segmentos de hasta bptt_len pasos entrenables.

    Si burn_in_len>0, cada segmento (salvo el arranque de episodio) antepone
    hasta burn_in_len pasos previos marcados `_burn=True`: la GRU los propaga
    sin entrar al loss (R2D2-style burn-in).
    """
    if not samples:
        return []
    episodes: dict = {}
    for s in samples:
        episodes.setdefault(s.get("_ep", 0), []).append(s)
    segs = []
    burn_in_len = max(0, int(burn_in_len))
    for ep_samples in episodes.values():
        n = len(ep_samples)
        for start in range(0, n, bptt_len):
            end = min(start + bptt_len, n)
            b_start = max(0, start - burn_in_len)
            n_burn = start - b_start
            seg = []
            for i, s in enumerate(ep_samples[b_start:end]):
                if i < n_burn:
                    ns = dict(s)
                    ns["_burn"] = True
                    seg.append(ns)
                else:
                    seg.append(s)
            if seg:
                segs.append(seg)
    return segs


def _seg_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Media sobre T de cada segmento. x, valid [B, T] → [B]."""
    w = valid.float()
    return (x * w).sum(dim=-1) / w.sum(dim=-1).clamp(min=1.0)


def _pad_targets(mb: list, T: int, device: str):
    """lp_old / adv / ret / v_old [B, T] (constantes, sin grafo)."""
    B = len(mb)
    lp_old = torch.zeros(B, T, device=device)
    adv = torch.zeros(B, T, device=device)
    ret = torch.zeros(B, T, device=device)
    v_old = torch.zeros(B, T, device=device)
    for b, seg in enumerate(mb):
        for t, s in enumerate(seg):
            lp = s["action"]["log_prob"]
            if torch.is_tensor(lp):
                lp_old[b, t] = lp.reshape(-1)[0].to(device)
            else:
                lp_old[b, t] = float(lp)
            adv[b, t] = s["adv"] if torch.is_tensor(s["adv"]) else float(s["adv"])
            ret[b, t] = s["ret"] if torch.is_tensor(s["ret"]) else float(s["ret"])
            vp = s["value_pred"]
            v_old[b, t] = vp if torch.is_tensor(vp) else float(vp)
    return lp_old, adv, ret, v_old


class PPOTrainer:
    def __init__(self, net, lr: float = 3e-4, device: str = "cpu",
                 clip_eps: float = 0.2, vf_coef: float = 0.5,
                 ent_lo: float = 0.01, ent_hi: float = 0.04,
                 max_grad_norm: float = 0.5, bptt_len: int = 32,
                 burn_in_len: int = 0):
        self.net = net.to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.device = device
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_lo = ent_lo
        self.ent_hi = ent_hi
        self.max_grad_norm = max_grad_norm
        self.bptt_len = bptt_len  # longitud de segmento para BPTT truncado
        self.burn_in_len = max(0, int(burn_in_len))
        self.scaler = _make_scaler(device)
        self.use_amp = device == "cuda"
        if self.use_amp:
            torch.backends.cudnn.benchmark = True

    def _eval_mb(self, mb: list):
        """lp, ent, val, valid [B, T]. Fallback por-seg para nets de test."""
        fn = getattr(self.net, "evaluate_actions_seq_batch", None)
        if callable(fn):
            return fn(mb, self.device)
        T = max(len(s) for s in mb)
        lps, ents, vals, masks = [], [], [], []
        for seg in mb:
            l, e, v = self.net.evaluate_actions_seq(seg, self.device)
            l = l.reshape(-1)
            e = e.reshape(-1)
            v = v.reshape(-1)
            n = l.numel()
            pad = T - n
            if pad:
                z = l.new_zeros(pad)
                l = torch.cat([l, z])
                e = torch.cat([e, e.new_zeros(pad)])
                v = torch.cat([v, v.new_zeros(pad)])
            valid = torch.zeros(T, dtype=torch.bool, device=l.device)
            valid[:n] = True
            lps.append(l)
            ents.append(e)
            vals.append(v)
            masks.append(valid)
        return (torch.stack(lps), torch.stack(ents),
                torch.stack(vals), torch.stack(masks))

    def _opt_step(self, loss) -> float:
        """AMP scale + clip. Devuelve grad_norm (0 si skip)."""
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        gn = torch.nn.utils.clip_grad_norm_(
            self.net.parameters(), self.max_grad_norm).item()
        if not math.isfinite(gn):
            self.net.zero_grad(set_to_none=True)
            self.scaler.update()
            return 0.0
        self.scaler.step(self.opt)
        self.scaler.update()
        return gn

    def update(self, samples: list, epochs: int = 2, batch_size: int = 32):
        """PPO recurrente con BPTT truncado por segmentos.

        Prefetch a GPU una vez + BPTT batcheado (B segmentos en encode) + AMP.
        El shuffle sigue siendo a nivel de segmento.
        """
        if not samples:
            return {}
        prefetch_steps(samples, self.device, inplace=True)
        segs = _split_segments(samples, self.bptt_len, self.burn_in_len)
        if not segs:
            return {}

        segs_per_batch = max(1, int(round(batch_size / max(1, self.bptt_len + self.burn_in_len))))
        stats = {"pi_loss": [], "v_loss": [], "entropy": [],
                 "clip_frac": [], "kl": []}
        gn = 0.0
        try:
            gn = self._ppo_epochs(segs, epochs, segs_per_batch, stats)
        except torch.cuda.OutOfMemoryError:
            if segs_per_batch <= 1:
                raise
            print(f"[update] OOM BPTT B={segs_per_batch} — retry B=1",
                  flush=True)
            torch.cuda.empty_cache()
            gn = self._ppo_epochs(segs, epochs, 1, stats)

        adv_vals = []
        for s in samples:
            v = s["adv"]
            adv_vals.append(float(v.reshape(-1)[0]) if torch.is_tensor(v)
                            else float(v))
        out = {k: round(float(np.mean(v)), 5) if v else 0.0
               for k, v in stats.items()}
        out |= {"grad_norm": round(gn, 4),
                "adv_mean": round(float(np.mean(adv_vals)), 5) if adv_vals else 0.0,
                "n": len(samples)}
        return out

    def _ppo_epochs(self, segs, epochs, segs_per_batch, stats) -> float:
        seg_idx = np.arange(len(segs))
        gn = 0.0
        for _ in range(epochs):
            np.random.shuffle(seg_idx)
            for start in range(0, len(segs), segs_per_batch):
                mb = [segs[i] for i in seg_idx[start:start + segs_per_batch]]
                self.net.zero_grad(set_to_none=True)
                with _autocast(self.device):
                    lp_new, entropy, value, valid = self._eval_mb(mb)
                lp_new = lp_new.float()
                entropy = entropy.float()
                value = value.float()
                T = lp_new.size(1)
                lp_old, adv, ret, v_old = _pad_targets(mb, T, self.device)
                finite = ((~valid)
                          | (torch.isfinite(lp_new)
                             & torch.isfinite(entropy)
                             & torch.isfinite(value)
                             & torch.isfinite(lp_old)))
                seg_ok = finite.all(dim=-1) & valid.any(dim=-1)
                if not bool(seg_ok.any()):
                    continue
                log_ratio = (lp_new - lp_old).clamp(
                    -_LOG_RATIO_CLAMP, _LOG_RATIO_CLAMP)
                ratio = torch.exp(log_ratio)
                if not torch.isfinite(ratio).all():
                    continue
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                    1 + self.clip_eps) * adv
                pi_t = -torch.min(surr1, surr2)
                pi_b = _seg_mean(pi_t, valid)
                v_clipped = v_old + torch.clamp(
                    value - v_old, -self.clip_eps, self.clip_eps)
                v1 = _seg_mean(
                    F.smooth_l1_loss(value, ret, reduction="none"), valid)
                v2 = _seg_mean(
                    F.smooth_l1_loss(v_clipped, ret, reduction="none"),
                    valid)
                v_b = torch.maximum(v1, v2)
                h_b = _seg_mean(entropy, valid)
                h_mean = float(h_b[seg_ok].mean().item())
                coef = self.ent_lo + self.ent_hi * max(
                    0.0, 1.0 - h_mean / 2.0)
                if h_mean < 0.5:
                    coef = max(coef, 0.15)
                # /len(mb): misma escala que el loop viejo (seg skipped = 0)
                seg_loss = pi_b + self.vf_coef * v_b - coef * h_b
                loss = seg_loss[seg_ok].sum() / len(mb)
                if not torch.isfinite(loss):
                    continue
                gn = self._opt_step(loss)
                if gn == 0.0:
                    continue
                with torch.no_grad():
                    w = valid.float()
                    clip_frac = (
                        ((ratio - 1).abs() > self.clip_eps).float() * w
                    ).sum() / w.sum().clamp(min=1.0)
                    kl = ((lp_new - lp_old).clamp(min=0) * w).sum() / w.sum().clamp(min=1.0)
                    stats["pi_loss"].append(float(pi_b[seg_ok].mean().item()))
                    stats["v_loss"].append(float(v_b[seg_ok].mean().item()))
                    stats["entropy"].append(h_mean)
                    stats["clip_frac"].append(float(clip_frac.item()))
                    stats["kl"].append(float(kl.item()))
        return gn

    def imitation_update(self, samples: list, coef: float,
                         epochs: int = 1, batch_size: int = 128) -> float:
        """NLL de acciones élite / maestro (BC y SIL). No usa advantages."""
        if not samples or coef <= 0.0:
            return 0.0
        # Copia a GPU; el EliteBuffer se queda en CPU.
        gpu_steps = prefetch_steps(samples, self.device, inplace=False)
        segs = _split_segments(gpu_steps, self.bptt_len, self.burn_in_len)
        if not segs:
            return 0.0
        segs_per_batch = max(1, int(round(batch_size / max(1, self.bptt_len + self.burn_in_len))))
        nlls = []
        try:
            nlls = self._sil_epochs(segs, coef, epochs, segs_per_batch)
        except torch.cuda.OutOfMemoryError:
            if segs_per_batch <= 1:
                raise
            print(f"[sil] OOM BPTT B={segs_per_batch} — retry B=1", flush=True)
            torch.cuda.empty_cache()
            nlls = self._sil_epochs(segs, coef, epochs, 1)
        return round(float(np.mean(nlls)), 5) if nlls else 0.0

    def _sil_epochs(self, segs, coef, epochs, segs_per_batch) -> list:
        nlls = []
        idx = np.arange(len(segs))
        for _ in range(epochs):
            np.random.shuffle(idx)
            for start in range(0, len(segs), segs_per_batch):
                mb = [segs[i] for i in idx[start:start + segs_per_batch]]
                self.net.zero_grad(set_to_none=True)
                with _autocast(self.device):
                    lp, _, _, valid = self._eval_mb(mb)
                lp = lp.float()
                finite = ((~valid) | torch.isfinite(lp))
                seg_ok = finite.all(dim=-1) & valid.any(dim=-1)
                if not bool(seg_ok.any()):
                    continue
                # Dest ilegal (agua) daba logit -1e9 → sil_nll ~7e6 (Run 13).
                lp = lp.clamp(min=-20.0)
                nll_b = -_seg_mean(lp, valid)
                keep = seg_ok & torch.isfinite(nll_b) & (
                    nll_b < _SIL_NLL_SKIP)
                if not bool(keep.any()):
                    continue
                # /len(mb): skipped no aportan (floor clamp 923).
                loss = nll_b[keep].sum() / len(mb)
                if not torch.isfinite(loss):
                    continue
                gn = self._opt_step(coef * loss)
                if gn == 0.0:
                    continue
                nlls.append(float(loss.item()))
        return nlls


def save_checkpoint(path: str, net, opt, iteration: int, extra: dict | None = None):
    ckpt = {"net": net.state_dict(), "opt": opt.state_dict(),
            "iteration": iteration, "time": time.time()}
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path: str, net, opt=None, vocab=None, reset_opt=False,
                    extra_out: dict | None = None):
    """Restaura red/opt/iteración y — si el checkpoint lo trae — el VOCAB.

    F2 (auditoría 2026-08-24): antes se ignoraba ckpt["vocab"] y cada resume
    reconstruía el vocabulario en orden de aparición → los ids de la cabeza
    de ítems podían quedar BARAJADOS respecto a los pesos guardados (parte
    del fracaso del incentivo minero tras los reinicios del ritual).

    reset_opt: leave Adam at fresh init (collapse restore). The moments of a
    dead policy keep the type-head pinned even after weights are replaced.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    from rl.network import adapt_capa2_state_dict, adapt_capa2c_state_dict, adapt_scalar_state_dict
    raw = ckpt["net"]
    adapted = adapt_scalar_state_dict(
        net, adapt_capa2c_state_dict(net, adapt_capa2_state_dict(net, raw)))
    incompat = net.load_state_dict(adapted, strict=False)
    n_miss = len(incompat.missing_keys)
    n_unex = len(incompat.unexpected_keys)
    # cell_head puede ser Conv2d (legacy) o Sequential (arch v1.1); no asumir .weight
    from rl.network import cell_head_weight_shape
    old_cell = raw.get("cell_head.weight")
    if old_cell is None:
        old_cell = raw.get("cell_head.2.weight")  # Sequential: ultimo conv
    old_shape = tuple(old_cell.shape) if old_cell is not None else ()
    new_shape = cell_head_weight_shape(net.cell_head)
    arch_changed = (
        n_miss > 0 or n_unex > 0
        or (bool(old_shape) and bool(new_shape) and old_shape != new_shape)
    )
    if n_miss or n_unex:
        print(f"[ckpt] Capa 2c Net2Net missing={n_miss} unexpected={n_unex} "
              f"(role_emb / mlp pad; tronco A)", flush=True)
    do_reset = bool(reset_opt or ckpt.get("reset_opt") or arch_changed)
    if opt is not None and "opt" in ckpt and not do_reset:
        try:
            opt.load_state_dict(ckpt["opt"])
        except (RuntimeError, ValueError) as e:
            print(f"[ckpt] Adam fresco (opt no carga: {e})", flush=True)
            do_reset = True
    if do_reset:
        print("[ckpt] Adam fresco (reset-opt)", flush=True)
    if vocab is not None and isinstance(ckpt.get("vocab"), dict):
        vocab.type_to_id = dict(ckpt["vocab"])
    if extra_out is not None:
        extra_out.clear()
        if ckpt.get("bc_start_iter") is not None:
            extra_out["bc_start_iter"] = int(ckpt["bc_start_iter"])
    return ckpt.get("iteration", 0)
