"""Rollout: recolecta trayectorias contra el server OpenRA-RL.

Diseño v0:
    - Un async worker por sesión del server (el runner soporta 64/proceso)
    - Frame-skip K: la acción elegida se mantiene K observaciones
      (mismo K en entrenamiento e inferencia)
    - old_log_prob y value se guardan CONGELADOS para PPO
    - hidden del GRU desacoplado del gradiente entre decisiones
    - GAE(λ) por trayectoria + centrado por grupo (por episodio)

Uso:
    episodes, outcomes = asyncio.run(collect_episodes(url, net, vocab, device))
"""

import asyncio
import time

import numpy as np
import torch

from openra_env.client import OpenRAEnv
from openra_env.models import ActionType, CommandModel, OpenRAAction
from rl.action_adapter import ActionIndex, Vocab, apply_passability, index_to_command_effective
from rl.imitation import command_to_indices, pick_bc_command
from rl.network import ACTION_TYPES, HIDDEN_DIM
from rl.obs_encoding import MAX_UNITS, decode_spatial, scalar_features, unit_slots
from rl.reward_shaping import PRESETS, ShapedReward
from rl.supremacy import evaluate_supremacy
from rl.economy_race import EconomyRace
from rl.auto_support import apply_dest_credit, support_commands


def _batch_of(obs, vocab, device):
    """Observación del env -> dict de tensores para la red (+ActionIndex)."""
    h = max(obs.map_info.height, 1)
    w = max(obs.map_info.width, 1)
    # No pintar (95,11) en Ch7/Ch8: es el spawn NE, no “el enemigo”.
    # Spawn invertido olía a casa. Ch7/Ch8 reales (visión) bastan.
    spatial = decode_spatial(obs.spatial_map, h, w, obs.spatial_channels or 9,
                             beacon=None)
    if spatial is None:
        spatial = np.zeros((9, h, w), dtype=np.float32)

    units_feats, unit_valid = unit_slots(obs)
    if len(units_feats) == 0:
        units_feats = np.zeros((0, 10), dtype=np.float32)
        unit_valid = np.zeros(0, dtype=bool)
    pad = MAX_UNITS - units_feats.shape[0]
    if pad > 0:
        units_feats = np.vstack([units_feats,
                                 np.zeros((pad, 10), dtype=np.float32)])
        unit_valid = np.concatenate([unit_valid, np.zeros(pad, dtype=bool)])

    aidx = ActionIndex(obs, vocab)
    # Channel 3 is passability (0/1). Mask illegal cells in the cell head so
    # attack_move cannot sample south-water (y≳40 on Singles). Fallback inside
    # apply_passability keeps all-true if the grid is empty (shell map).
    apply_passability(aidx, spatial[3])
    return {
        "spatial": torch.from_numpy(spatial).unsqueeze(0).to(device),
        "scalars": torch.from_numpy(scalar_features(obs)).unsqueeze(0).to(device),
        "unit_feats": torch.from_numpy(units_feats).unsqueeze(0).to(device),
        "unit_valid": torch.from_numpy(unit_valid).unsqueeze(0).to(device),
        "type_mask": aidx.type_mask.unsqueeze(0).to(device),
        "cell_mask": aidx.cell_mask.unsqueeze(0).to(device),
        "item_indices": aidx.item_indices.unsqueeze(0).to(device),
        "item_mask": aidx.item_mask.unsqueeze(0).to(device),
        "train_slot_mask": aidx.train_slot_mask.unsqueeze(0).to(device),
        "build_slot_mask": aidx.build_slot_mask.unsqueeze(0).to(device),
    }, aidx


async def collect_one_episode(env: OpenRAEnv, net, vocab: Vocab, device: str,
                              k_skip: int = 8, max_steps: int = 4000,
                              temperature: float = 1.0,
                              telemetry: list | None = None,
                              macro_ticks: int = 0,
                              reset_kwargs: dict | None = None,
                              shaper_preset: str = "eradicate",
                              auto_support: bool = False,
                              teacher=None):
    """Juega UNA partida completa; devuelve (trayectoria, resumen).

    max_steps limita los env.step (cada uno avanza 2 ticks del juego):
    4000 pasos ≈ 8000 ticks ≈ 5.3 min de juego.

    macro_ticks > 0 activa el modo MACRO (v4): cada iteración es UNA decisión
    y max_steps cuenta DECISIONES. Tras emitir comandos (step +2 ticks) se
    completa el presupuesto del bloque con env.advance() (avance acelerado
    con interrupciones server-side: enemy_spotted, production_complete, etc.)
    y se cierra con step(NO_OP) para refrescar la observación completa.

    telemetry: lista opcional donde se vuelca un resumen por decisión
    (para análisis offline sin abrir el juego ni usar visión).
    """
    use_macro = macro_ticks > 0
    result = await env.reset(**(reset_kwargs or {}))
    obs = result.observation

    hidden = torch.zeros(1, HIDDEN_DIM, device=device)
    traj = []
    episode_reward = 0.0
    t0 = time.time()
    pending_cmd = None   # comando re-aplicado durante el frame-skip
    pending_sample = None
    ep_dims = None       # dims del mapa del episodio (las primeras obs pueden
                         # venir con dims del shell map mientras inicializa)
    dims_hist = {}       # histograma de dims vistos; la moda = mapa confiable
    if shaper_preset not in PRESETS:
        raise ValueError(f"shaper_preset desconocido: {shaper_preset!r}")
    shaper = ShapedReward(preset=shaper_preset)
    shaper.reset(obs)
    consec_errors = 0
    outcome_error = False
    done = False
    macro_final_result = None
    interrupt_reason = None
    advanced_total = 0   # ticks avanzados vía advance() (modo macro)
    interrupts = {}      # conteo por razón de interrupción
    # Carrera económica: riqueza por bando muestreada durante el episodio.
    # Muestra inicial desde la obs local (el enemigo es niebla aún).
    race = EconomyRace()
    race.add(obs.tick,
             own_cash=getattr(obs.economy, "cash", 0),
             own_wealth=(getattr(obs.economy, "cash", 0)
                         + getattr(obs.economy, "ore", 0)
                         + getattr(obs.military, "assets_value", 0)))
    # Histograma de tipos de acción EFECTIVOS del episodio (qué hace la
    # política: train/no_op vs remate army_attack_move) + conteo espectador
    # de edificios de cada bando al cierre (baja al razer).
    action_counts = {}
    own_n_buildings = 0
    ene_n_buildings = 0
    last_push_cell = None  # (x, y) del último army/attack_move

    for step in range(max_steps):
        # Decidir SIEMPRE cada k_skip (o cada iteración en modo macro):
        # las obs iniciales pueden traer dims del shell map; esas muestras
        # basura las descarta el filtro de dims mayoritario en train.py.
        can_decide = use_macro or step % k_skip == 0

        if can_decide:
            batch, aidx = _batch_of(obs, vocab, device)
            cur_dims = (obs.map_info.height, obs.map_info.width)
            dims_hist[cur_dims] = dims_hist.get(cur_dims, 0) + 1
            ep_dims = max(dims_hist, key=dims_hist.get)  # moda = mapa real
            h_in = hidden.detach().clone()
            had_item = aidx.item_mask.any().view(1).to(device)
            if teacher is not None:
                with torch.no_grad():
                    _fmap, _, hidden, _tok = net.encode(
                        batch["spatial"], batch["scalars"],
                        batch["unit_feats"], batch["unit_valid"], hidden)
                    hidden = hidden.detach()
                    value_t = net.value_head(hidden).squeeze(-1)
                raw = teacher.decide(obs)
                primary = pick_bc_command(raw.commands)
                t0, u0, c0, i0 = command_to_indices(obs, primary, aidx)
                action, (eff_t, eff_u, eff_i, eff_c) = index_to_command_effective(
                    obs, t0, u0, c0, i0, aidx)
                extras = [c for c in (raw.commands or []) if c is not primary]
                action.commands.extend(extras)
                sampled = (t0, u0, i0, c0)
                out_type = t0
                out_unit = u0
                out_item = i0
                out_cell = c0
                out_value = float(value_t.item())
                cell_t = torch.tensor([int(eff_c)], device=device)
                with torch.no_grad():
                    log_prob, _, _ = net.evaluate_actions(
                        batch, h_in, {
                            "type": torch.tensor([eff_t], device=device),
                            "unit_slot": torch.tensor([eff_u], device=device),
                            "cell_flat": cell_t,
                            "item_slot": torch.tensor([eff_i], device=device),
                            "had_item": had_item,
                        })
            else:
                out = net.act(batch, hidden, temperature=temperature)
                hidden = out["hidden"].detach()
                action, (eff_t, eff_u, eff_i, eff_c) = index_to_command_effective(
                    obs, int(out["type"]), int(out["unit_slot"]),
                    int(out["cell_flat"]), int(out["item_slot"]), aidx,
                )
                sampled = (int(out["type"]), int(out["unit_slot"]),
                           int(out["item_slot"]), int(out["cell_flat"]))
                out_type = int(out["type"])
                out_unit = int(out["unit_slot"])
                out_item = int(out["item_slot"])
                out_cell = int(out["cell_flat"])
                out_value = float(out["value"].item())
                log_prob = out["log_prob"]
                cell_t = out["cell_flat"]
            # Crédito de dest (Capa 0/1, no Capa 2): army/attack_move entra al
            # buffer con el dest de auto_support, no el sample en casa (Ch6).
            # Mutar el comando + cell_flat; F1 abajo recálcula log π.
            if auto_support:
                new_c, _dest_xy = apply_dest_credit(
                    obs, action, ACTION_TYPES[eff_t], int(eff_c), aidx,
                    last_push=last_push_cell)
                if int(new_c) != int(eff_c):
                    eff_c = int(new_c)
                    cell_t = torch.tensor([int(eff_c)], device=device)
            # F1 coerción COMPLETA (auditoría 2026-08-24): si una corrección
            # de seguridad MUTÓ la acción, recalcular log π(a_ejecutada|s)
            # con h_in — la MISMA semilla de hidden con la que se muestreó —
            # y guardar en el buffer los ÍNDICES EFECTIVOS.
            # La versión anterior tenía 3 bugs: (1) recalculaba con hidden
            # post-act (otra distribución temporal), (2) guardaba los índices
            # MUESTREADOS junto al log_prob efectivo (el ratio de PPO
            # comparaba acciones distintas), (3) solo cubría mutaciones con
            # ítem (attack→attack_move, harvest, deploy quedaban fuera).
            effective = (eff_t, eff_u, eff_i, int(eff_c))
            if int(cell_t) != int(eff_c):
                cell_t = torch.tensor([int(eff_c)], device=device)
            if sampled != effective:
                with torch.no_grad():
                    re_lp, _, _ = net.evaluate_actions(
                        batch, h_in, {
                            "type": torch.tensor([eff_t], device=device),
                            "unit_slot": torch.tensor([eff_u], device=device),
                            "cell_flat": cell_t if torch.is_tensor(cell_t) else
                            torch.tensor([int(eff_c)], device=device),
                            "item_slot": torch.tensor([eff_i], device=device),
                            "had_item": had_item,
                        })
                    log_prob = re_lp
            # Histograma de tipos de acción EFECTIVOS (traduce la política:
            # train/no_op vs el remate army_attack_move que nunca usa).
            atype = ACTION_TYPES[eff_t]
            action_counts[atype] = action_counts.get(atype, 0) + 1
            if telemetry is not None:
                telemetry.append({
                    "step": step,
                    "tick": obs.tick,
                    "type": ACTION_TYPES[int(out_type)],
                    "unit_id": (aidx.unit_ids[int(out_unit)]
                                if int(out_unit) < len(aidx.unit_ids)
                                else None),
                    "cell": [int(out_cell) % aidx.w,
                             int(out_cell) // aidx.w],
                    "item": (aidx.items[int(out_item)]
                             if int(out_item) < len(aidx.items)
                             else None),
                    "n_units": len(obs.units),
                    "cash": obs.economy.cash,
                })
            # Clamp de coordenadas al mapa real (la obs puede ser del shell)
            for c in action.commands:
                if c.target_x >= ep_dims[1] or c.target_y >= ep_dims[0]:
                    c.target_x = min(c.target_x, ep_dims[1] - 1)
                    c.target_y = min(c.target_y, ep_dims[0] - 1)
            # Destino de push vivo: si esta decisión fue army/attack_move,
            # los ociosos de los próximos bloques siguen hacia esa celda.
            if atype in ("army_attack_move", "attack_move") and action.commands:
                c0 = action.commands[0]
                if getattr(c0, "target_x", None) is not None:
                    last_push_cell = (int(c0.target_x), int(c0.target_y))
            # Pilar B: autonomía de soporte (0 decisiones, gratis para PPO)
            if auto_support:
                for cmd in support_commands(obs, last_push=last_push_cell, aidx=aidx):
                    action.commands.append(cmd)
            pending_cmd = action
            # F1: al buffer van los ÍNDICES EFECTIVOS (los de la acción que
            # realmente se ejecutó), emparejados con SU log_prob. El ratio de
            # PPO compara así π_nuevo(a_ejecutada) / π_viejo(a_ejecutada).
            pending_sample = {
                "batch": {k: v.cpu() for k, v in batch.items()},
                "action": {
                    "type": torch.tensor([effective[0]]),
                    "unit_slot": torch.tensor([effective[1]]),
                    "cell_flat": (cell_t.detach().cpu() if torch.is_tensor(cell_t)
                                  else torch.tensor([int(eff_c)])),
                    "item_slot": torch.tensor([effective[2]]),
                    "had_item": had_item.cpu(),
                    # CONGELADOS: la referencia contra la que PPO mide el drift
                    "log_prob": log_prob.detach().cpu(),
                },
                "reward": 0.0,
                "value_pred": out_value,
                "h_in": h_in.cpu(),
            }

        result = None
        try:
            result = await env.step(pending_cmd)
        except RuntimeError as e:
            # Crash del handler C# (ej. atacar un actor que murió entre la
            # obs y la ejecución — solo pasa con combate activo). Degradamos
            # el paso a NO_OP; si se repite demasiado, abortamos el episodio
            # conservando la trayectoria parcial.
            consec_errors += 1
            print(f"  [engine] step {step}: {str(e)[:120]} "
                  f"(errores consecutivos: {consec_errors})")
            if consec_errors >= 5:
                outcome_error = True
                break
            try:
                result = await env.step(OpenRAAction(
                    commands=[CommandModel(action=ActionType.NO_OP)]))
            except RuntimeError as e2:
                # El paso de RECUPERACIÓN también puede chocar contra el
                # mismo handler roto (dos crashes seguidos mataban el run
                # completo — expuesto por v4-macro con combate activo).
                consec_errors += 1
                print(f"  [engine] recovery {step}: {str(e2)[:120]} "
                      f"(errores consecutivos: {consec_errors})")
                if consec_errors >= 5:
                    outcome_error = True
                    break
                continue  # sin obs nueva: reintentar decisión sobre la última
        consec_errors = 0
        obs = result.observation
        # Reward CONFORMADO del lado del agente: se ACUMULA en la muestra
        # vigente porque los deltas de combate/economía pueden ocurrir en
        # cualquier frame del skip window, no solo en el de decisión.
        r_frame = shaper.step(obs, done=bool(result.done),
                               gs=getattr(race, "_last_gs", None),
                               action_type=atype)
        done = result.done
        episode_reward += r_frame
        if pending_sample is not None:
            pending_sample["reward"] = pending_sample.get("reward", 0.0) + r_frame

        # ── MODO MACRO (v4): completar el bloque con avance acelerado ──
        # Tras step(comandos) (+2 ticks), avanzar macro_ticks-2 más vía el
        # tool advance() (ráfagas de ≤50 ticks con interrupciones automáticas)
        # y cerrar con step(NO_OP): trae observación COMPLETA (con military,
        # que advance no devuelve) para la próxima decisión y para cerrar los
        # deltas del shaper del bloque.
        interrupt_reason = None
        macro_final_result = None
        if use_macro and not done:
            restante = max(0, macro_ticks - 2)
            try:
                while restante > 0 and not done:
                    adv = await env.advance(min(50, restante))
                    advanced_total += int(adv.get("actual_ticks_advanced", 0) or 0)
                    done = bool(adv.get("done", False))
                    if done:
                        macro_final_result = adv.get("result") or None
                    # Carrera económica: cada ráfaga trae el resumen exacto
                    gs = adv.get("global_summary")
                    if isinstance(gs, dict):
                        race._last_gs = gs
                        race.add_global_summary(adv.get("tick", 0), gs)
                        ene_g = gs.get("enemy") or {}
                        own_g = gs.get("own") or {}
                        ene_n_buildings = int(ene_g.get("n_buildings", 0) or 0)
                        own_n_buildings = int(own_g.get("n_buildings", 0) or 0)
                    if adv.get("interrupted"):
                        interrupt_reason = adv.get("interrupt_reason") or "?"
                    restante -= int(adv.get("actual_ticks_advanced", 0) or 0)
                if not done:
                    # Cierre del bloque: observación COMPLETA para la próxima
                    # decisión Y para moldear los deltas acumulados del bloque
                    # (advance() no trae military: sin este paso, todo lo
                    # pasado dentro del bloque no pagaría nada).
                    result = await env.step(OpenRAAction(
                        commands=[CommandModel(action=ActionType.NO_OP)]))
                    obs = result.observation
                    done = bool(result.done)
                    last_gs = getattr(race, "_last_gs", None)
                    r_close = shaper.step(obs, done=done, gs=last_gs, action_type=atype, closing=True)
                    episode_reward += r_close
                    if pending_sample is not None:
                        pending_sample["reward"] = \
                            pending_sample.get("reward", 0.0) + r_close
                    # Muestra de cierre del bloque: obs completa con military.
                    # El enemigo sigue siendo niebla aquí (solo visible), así
                    # que la riqueza rival viene de la última ráfaga exacta.
                    own_w = (getattr(obs.economy, "cash", 0)
                             + getattr(obs.economy, "ore", 0)
                             + getattr(obs.military, "assets_value", 0))
                    ene_w = None
                    if last_gs:
                        ene = last_gs.get("enemy") or {}
                        ene_w = (ene.get("cash", 0) + ene.get("unit_value", 0)
                                 + ene.get("building_value", 0))
                    race.add(obs.tick,
                             own_cash=getattr(obs.economy, "cash", 0),
                             own_wealth=own_w, enemy_wealth=ene_w)
            except Exception as e:
                msg = str(e)
                is_deadline = "DEADLINE" in msg or "Deadline" in msg
                # DEADLINE_EXCEEDED = sesión envenenada (World.Tick() colgado).
                # El retry sobre el mismo session_id solo reproduce el cuelgue.
                # Romper rápido, marcar el episodio como timeout y dejar el
                # env en estado que el caller (train.py::worker) saneará con
                # un reset fresco en el próximo episodio del pool.
                if is_deadline:
                    print(f"  [engine] advance {step}: DEADLINE_EXCEEDED "
                          f"(ticks={macro_ticks}, batalla grande) — "
                          f"sesión envenenada, abortando episodio")
                    outcome_error = True
                    # Best-effort: intentar cerrar la sesión colgada sin
                    # bloquear el rollout; el próximo reset del pool la recrea.
                    try:
                        # destroy es best-effort; el daemon la GCea si no responde
                        import grpc as _grpc
                        _ = _grpc  # evita unused-import linter
                        try:
                            # Env puede exponer destroy_session vía bridge; si no, no-op
                            br = getattr(env, "_bridge", None) or getattr(env, "bridge", None)
                            if br is not None and hasattr(br, "destroy_session"):
                                br.destroy_session()
                        except Exception:
                            pass
                    except Exception:
                        pass
                    break
                consec_errors += 1
                print(f"  [engine] advance {step}: {msg[:120]} "
                      f"(errores consecutivos: {consec_errors})")
                if consec_errors >= 5:
                    outcome_error = True
                    break
            else:
                consec_errors = 0

        if can_decide:
            traj.append(pending_sample)

        # F6 (auditoría): contar interrupciones SIEMPRE — antes solo se
        # registraban cuando pasaba telemetry (que train nunca pasa) y
        # metrics.jsonl decía "interrupts": {} desde hace cientos de iters.
        if interrupt_reason:
            interrupts[interrupt_reason] = interrupts.get(interrupt_reason, 0) + 1
        if use_macro and telemetry is not None and pending_sample is not None:
            telemetry[-1]["macro_ticks"] = macro_ticks
            telemetry[-1]["interrupt_reason"] = interrupt_reason

        if done:
            break

    # Cierre de episodio: win/lose viajan a finalize (pago único; step()
    # puede haberlo cobrado ya si done llegó dentro del bloque). En
    # eradicate el truncamiento paga 0 — el margen-al-cap era la muleta
    # que convertía "estar adelante al reloj" en óptimo local.
    r_final = shaper.finalize(
        truncated=not done,
        result=str(getattr(obs, "result", "") or macro_final_result or ""))
    episode_reward += r_final
    if traj and r_final != 0.0:
        traj[-1]["reward"] += r_final

    # F5 (auditoría): GAE necesita next_value — 0 en terminal real, V(s')
    # al truncar. Antes add_advantages usaba values[-1] (el valor del PROPIO
    # último estado) → δ_T = r_T y el crítico jamás vio "valor restante".
    # Se computa acá porque solo el rollout tiene la obs final + hidden.
    if traj and not done and not outcome_error:
        with torch.no_grad():
            b_final, _ = _batch_of(obs, vocab, device)
            _, _, h_fin, _tok = net.encode(
                b_final["spatial"], b_final["scalars"],
                b_final["unit_feats"], b_final["unit_valid"], hidden)
            traj[-1]["_v_next"] = float(
                net.value_head(h_fin).squeeze(-1).item())

    final_result = ("engine_error" if outcome_error
                    else getattr(obs, "result", None)
                    or macro_final_result
                    or ("incomplete" if not obs.done else ""))
    # La obs del agente no expone global_summary (solo la niebla), por eso la
    # supremacía salía inflada a favor (enemigo ~0). Inyectarle el último
    # global_summary que devolvió advance() para evaluar con el espectador
    # exacto. Si no hay (no-macro), remain a niebla como antes.
    gs_exacto = getattr(race, "_last_gs", None)
    if isinstance(gs_exacto, dict) and gs_exacto.get("own") and gs_exacto.get("enemy"):
        try:
            obs.global_summary = gs_exacto
        except Exception:
            pass
    outcome = {
        "result": final_result,
        "ticks": obs.tick,
        "decisions": len(traj),
        "episode_reward": round(episode_reward, 3),
        "wall_s": round(time.time() - t0, 1),
        # Descomposición del reward por componente (combate/economía/etc):
        # permite distinguir 'reward ciego' de colapso de política en las
        # métricas sin abrir el juego.
        "reward_components": {k: round(v, 4)
                              for k, v in shaper.last_components.items()},
        # Modo macro: cuántos ticks se recorrieron vía advance()
        "advanced_ticks": advanced_total,
        # Razones de interrupción del bloque ({razón: cantidad})
        "interrupts": interrupts,
        # Supremacía final estilo motor de ajedrez (nivel 1: score estático)
        "supremacy": evaluate_supremacy(obs, gs=gs_exacto),
        # Nivel 2: curva de evaluación del crítico V(s) por decisión
        # (la 'barra de Lichess': cómo percibió el crítico la partida)
        "value_curve": [round(float(s["value_pred"]), 3) for s in traj],
        # Carrera económica: agregados por episodio (pendientes /1000 ticks)
        "economy_race": race.resumen(),
        # Series completas (van a archivo aparte en train.py, no a métricas)
        "economy_race_series": race.series(),
        # Histograma de tipos de acción efectivos del episodio
        # (para ver train/no_op vs el remate army_attack_move)
        "action_hist": dict(action_counts),
        # Conteo espectador de edificios de cada bando al cierre
        # (el rival baja cuando raseas su base)
        "n_buildings": {"own": own_n_buildings, "enemy": ene_n_buildings},
    }
    return traj, outcome


async def collect_episodes(url: str, net, vocab: Vocab, device: str,
                           n_episodes: int = 2, k_skip: int = 8,
                           temperature: float = 1.0, concurrency: int = 1):
    """Recolecta N episodios con concurrencia limitada."""
    sem = asyncio.Semaphore(concurrency)
    all_traj, outcomes = [], []

    async def one():
        async with sem:
            async with OpenRAEnv(base_url=url, message_timeout_s=300.0) as env:
                traj, outcome = await collect_one_episode(
                    env, net, vocab, device, k_skip=k_skip,
                    temperature=temperature)
                all_traj.append(traj)
                outcomes.append(outcome)

    await asyncio.gather(*(one() for _ in range(n_episodes)))
    return all_traj, outcomes


def add_advantages(traj: list, gamma: float = 0.99, lam: float = 0.95,
                   last_value: float | None = None):
    """GAE(λ) in-place sobre una trayectoria.

    F5 (auditoría 2026-08-24): si el último paso trae '_v_next' (valor de
    V(s') computado en el rollout al TRUNCAR), se usa como bootstrap —
    antes next_v caía en values[-1] → δ_T = r_T (pseudo-terminal) y el
    crítico nunca vio valor restante. last_value explícito tiene prioridad.
    """
    T = len(traj)
    if T == 0:
        return
    values = [s["value_pred"] for s in traj]
    if last_value is not None:
        nv_final = last_value
    else:
        nv_final = float(traj[-1].pop("_v_next", 0.0) or 0.0)
    next_v = nv_final
    gae = 0.0
    for t in reversed(range(T)):
        delta = traj[t]["reward"] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        traj[t]["adv"] = gae
        traj[t]["ret"] = gae + values[t]
        next_v = values[t]


def center_advantage_by_episode(episodes: list):
    """Centrado POR GRUPO (cada episodio = un grupo): resta la media del
    episodio. Sin división por std — con grupos chicos amplifica ruido
    numérico hasta explotar el gradiente (lección alto-truco).
    """
    for traj in episodes:
        if not traj:
            continue
        advs = np.array([s["adv"] for s in traj], dtype=np.float32)
        mean = advs.mean()
        for s in traj:
            s["adv"] = s["adv"] - mean


def flatten_samples(episodes: list) -> list:
    return [s for traj in episodes for s in traj]
