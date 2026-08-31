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


class PPOTrainer:
    def __init__(self, net, lr: float = 3e-4, device: str = "cpu",
                 clip_eps: float = 0.2, vf_coef: float = 0.5,
                 ent_lo: float = 0.01, ent_hi: float = 0.04,
                 max_grad_norm: float = 0.5, bptt_len: int = 32):
        self.net = net.to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.device = device
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_lo = ent_lo
        self.ent_hi = ent_hi
        self.max_grad_norm = max_grad_norm
        self.bptt_len = bptt_len  # longitud de segmento para BPTT truncado

    def update(self, samples: list, epochs: int = 2, batch_size: int = 32):
        """PPO recurrente con BPTT truncado por segmentos.

        Antes: las transiciones se aplanaban y se mezclaban al azar; el h_in
        se trataba como constante y el gradiente NUNCA viajaba h_t→h_{t+1}
        (memoria temporal rota). Ahora: se entrenan SEGMENTOS de hasta
        bptt_len pasos consecutivos del mismo episodio (_ep), propagando el
        GRU sin detach (evaluate_actions_seq) → la red aprende qué guardar
        en el hidden. El shuffle es a nivel de segmento, preservando la
        secuencia.
        """
        if not samples:
            return {}

        # 1) Construir segmentos: bloques de ≤ bptt_len pasos consecutivos,
        #    cortando cuando cambia el episodio (_ep).
        ep0 = samples[0].get("_ep", 0)
        segs, run = [], [samples[0]]
        for s in samples[1:]:
            if s.get("_ep", ep0) != run[0].get("_ep", ep0) or len(run) >= self.bptt_len:
                segs.append(run)
                run = [s]
            else:
                run.append(s)
        if run:
            segs.append(run)
        if not segs:
            return {}

        seg_idx = np.arange(len(segs))
        segs_per_batch = max(1, int(round(batch_size / self.bptt_len)))
        stats = {"pi_loss": [], "v_loss": [], "entropy": [],
                 "clip_frac": [], "kl": []}
        gn = 0.0

        for _ in range(epochs):
            np.random.shuffle(seg_idx)
            for start in range(0, len(segs), segs_per_batch):
                mb = [segs[i] for i in seg_idx[start:start + segs_per_batch]]

                # backward por segmento (acumula grad en params): 1 grafo a la
                # vez → memoria SÓLO del segmento actual, no de todo el batch.
                self.net.zero_grad(set_to_none=True)
                coef = None
                n_back = 0
                for seg in mb:
                    lp_new, entropy, value = self.net.evaluate_actions_seq(
                        seg, self.device)
                    if (not torch.isfinite(lp_new).all()
                            or not torch.isfinite(value).all()
                            or not torch.isfinite(entropy).all()):
                        continue
                    lp_old = torch.cat(
                        [s["action"]["log_prob"] for s in seg]).to(self.device)
                    if not torch.isfinite(lp_old).all():
                        continue
                    v_old = torch.tensor(
                        [s["value_pred"] for s in seg], device=self.device)
                    mb_adv = torch.tensor(
                        [s["adv"] for s in seg], device=self.device)
                    mb_ret = torch.tensor(
                        [s["ret"] for s in seg], device=self.device)

                    log_ratio = (lp_new - lp_old).clamp(
                        -_LOG_RATIO_CLAMP, _LOG_RATIO_CLAMP)
                    ratio = torch.exp(log_ratio)
                    if not torch.isfinite(ratio).all():
                        continue
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                        1 + self.clip_eps) * mb_adv
                    pi_loss = -torch.min(surr1, surr2).mean()

                    v_clipped = v_old + torch.clamp(value - v_old,
                                                    -self.clip_eps, self.clip_eps)
                    v_loss = torch.max(F.smooth_l1_loss(value, mb_ret),
                                       F.smooth_l1_loss(v_clipped, mb_ret))
                    entropy_mean = entropy.mean()
                    if coef is None:
                        h_mean = float(entropy_mean.item())
                        coef = self.ent_lo + self.ent_hi * max(
                            0.0, 1.0 - h_mean / 2.0)
                        # Near-collapse: the 0.05 ceiling cannot unstick a
                        # peaked type-head. Bump the bonus while H is still
                        # moving; H≈0 batches are skipped in train.py instead.
                        if h_mean < 0.5:
                            coef = max(coef, 0.15)
                    loss = ((pi_loss + self.vf_coef * v_loss
                             - coef * entropy_mean) / len(mb))
                    if (not torch.isfinite(pi_loss)
                            or not torch.isfinite(v_loss)
                            or not torch.isfinite(loss)):
                        continue
                    # divide por nº de segmentos para NO escalar el grad con
                    # len(mb)>1 (si no, la lr efectiva se duplica con 2 seg)
                    loss.backward()
                    n_back += 1

                    with torch.no_grad():
                        clip_frac = ((ratio - 1).abs() > self.clip_eps).float().mean()
                        kl = (lp_new - lp_old).mean().clamp(min=0)
                        stats["pi_loss"].append(float(pi_loss.item()))
                        stats["v_loss"].append(float(v_loss.item()))
                        stats["entropy"].append(float(entropy_mean.item()))
                        stats["clip_frac"].append(float(clip_frac.item()))
                        stats["kl"].append(float(kl.item()))

                if n_back == 0:
                    continue
                gn = torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.max_grad_norm).item()
                if not math.isfinite(gn):
                    self.net.zero_grad(set_to_none=True)
                    continue
                self.opt.step()

        out = {k: round(float(np.mean(v)), 5) if v else 0.0
               for k, v in stats.items()}
        out |= {"grad_norm": round(gn, 4),
                "adv_mean": round(float(np.mean(
                    [s["adv"] for s in samples])), 5), "n": len(samples)}
        return out

    def imitation_update(self, samples: list, coef: float,
                         epochs: int = 1, batch_size: int = 128) -> float:
        """NLL de acciones élite / maestro (BC y SIL). No usa advantages."""
        if not samples or coef <= 0.0:
            return 0.0
        ep0 = samples[0].get("_ep", 0)
        segs, run = [], [samples[0]]
        for s in samples[1:]:
            if s.get("_ep", ep0) != run[0].get("_ep", ep0) or len(run) >= self.bptt_len:
                segs.append(run)
                run = [s]
            else:
                run.append(s)
        if run:
            segs.append(run)
        if not segs:
            return 0.0
        segs_per_batch = max(1, int(round(batch_size / self.bptt_len)))
        nlls = []
        idx = np.arange(len(segs))
        for _ in range(epochs):
            np.random.shuffle(idx)
            for start in range(0, len(segs), segs_per_batch):
                mb = [segs[i] for i in idx[start:start + segs_per_batch]]
                self.net.zero_grad(set_to_none=True)
                loss = None
                for seg in mb:
                    lp, _, _ = self.net.evaluate_actions_seq(seg, self.device)
                    if not torch.isfinite(lp).all():
                        continue
                    # Dest ilegal (agua) daba logit -1e9 → sil_nll ~7e6 (Run 13).
                    lp = lp.clamp(min=-20.0)
                    nll = -lp.mean()
                    # Floor del clamp: la acción sigue tapada. No clonar (923).
                    if (not torch.isfinite(nll)
                            or float(nll.item()) >= _SIL_NLL_SKIP):
                        continue
                    nll = nll / len(mb)
                    loss = nll if loss is None else loss + nll
                if loss is None or not torch.isfinite(loss):
                    continue
                (coef * loss).backward()
                gn = torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.max_grad_norm).item()
                if not math.isfinite(gn):
                    self.net.zero_grad(set_to_none=True)
                    continue
                self.opt.step()
                nlls.append(float(loss.item()))
        return round(float(np.mean(nlls)), 5) if nlls else 0.0


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
    net.load_state_dict(ckpt["net"])
    do_reset = bool(reset_opt or ckpt.get("reset_opt"))
    if opt is not None and "opt" in ckpt and not do_reset:
        opt.load_state_dict(ckpt["opt"])
    elif do_reset:
        print("[ckpt] Adam fresco (reset-opt)", flush=True)
    if vocab is not None and isinstance(ckpt.get("vocab"), dict):
        vocab.type_to_id = dict(ckpt["vocab"])
    if extra_out is not None:
        extra_out.clear()
        if ckpt.get("bc_start_iter") is not None:
            extra_out["bc_start_iter"] = int(ckpt["bc_start_iter"])
    return ckpt.get("iteration", 0)
