"""Loop de entrenamiento PPO contra el server OpenRA-RL.

Arquitectura v0.2 (uso real del hardware):
    - Pool de conexiones WebSocket persistente (sin reconectar por iteración)
    - Episodios en PARALELO sobre el pool (el daemon .NET los simula en paralelo)
    - Update en thread + infer_net congelado: launch_collection de k+1 corre
      de verdad durante trainer.update(k). El event loop ya no se bloquea.
      collect usa una copia de pesos (1 paso stale; el clip PPO lo cubre).
    - Escalado de ventajas en UN solo lugar (process_results, según --adv-mode)

Requisitos: server corriendo (docker compose up openra-rl), PYTHONPATH limpio.

Uso:
    python -m rl.train --url http://localhost:8000 --iters 200 --episodes 8 \
        --concurrency 4 --device auto
"""

import argparse
import asyncio
import base64
import json
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from openra_env.client import OpenRAEnv
from rl.action_adapter import Vocab
from rl.reward_shaping import PRESETS as SHAPER_PRESETS
from rl.network import AlphaLiteNet
from rl.rollout import (add_advantages, center_advantage_by_episode,
                        collect_one_episode, flatten_samples)
from rl.trainer import PPOTrainer, load_checkpoint, save_checkpoint
from rl.best_ckpt import batch_is_dead, maybe_update_best
from rl.imitation import EliteBuffer, balance_bc_samples, lambda_bc_at
from rl.scripted_teacher import ScriptedTeacher


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def sample_dims(s):
    sp = s["batch"]["spatial"]
    return tuple(sp.shape[-2:])


def process_results(results, gamma, lam, verbose=True,
                    adv_mode: str = "episode"):
    """results: [(traj, outcome)] -> (samples filtrados, outcomes).

    adv_mode:
      - 'episode': centrado por grupo + división por la std del BATCH completo
        (comportamiento histórico: antes el trainer re-normalizaba siempre;
        F8 movió TODO el escalado acá — misma escala final, un solo lugar)
      - 'global': Z-score sobre TODO el batch (revisión externa: conserva
        la señal ENTRE episodios — el centrado por grupo la destruye cuando
        unas partidas son mejores que otras; lección alto-truco: NUNCA
        dividir por std de grupos chicos, por eso el global usa todo el
        batch de ~1200 muestras)
      - 'none': ventajas crudas de GAE sin tocar
    """
    episodes = [t for t, _ in results]
    outcomes = [o for _, o in results]
    for traj in episodes:
        add_advantages(traj, gamma=gamma, lam=lam)
    if adv_mode == "episode":
        center_advantage_by_episode(episodes)
        # F8: el escalado que antes hacía el trainer vive acá (centrado por
        # episodio + división por la std del BATCH COMPLETO ≈ escala histórica
        # exacta; ~1200 muestras, no grupos chicos).
        all_advs = np.concatenate(
            [np.array([s["adv"] for s in traj], dtype=np.float32)
             for traj in episodes if traj])
        sd = float(all_advs.std())
        if sd > 1e-8:
            for traj in episodes:
                for s in traj:
                    s["adv"] = s["adv"] / sd
    elif adv_mode == "global":
        all_advs = np.concatenate(
            [np.array([s["adv"] for s in traj], dtype=np.float32)
             for traj in episodes if traj])
        mu, sd = float(all_advs.mean()), float(all_advs.std())
        if sd > 1e-8:
            for traj in episodes:
                for s in traj:
                    s["adv"] = (s["adv"] - mu) / sd

    # Marcar cada muestra con su episodio (_ep) para que el entrenamiento por
    # segmentos (BPTT) no cruce entre partidas distintas.
    samples = []
    for ep_i, traj in enumerate(episodes):
        for s in traj or []:
            s["_ep"] = ep_i
            samples.append(s)
    if samples:
        dims_count = Counter(sample_dims(s) for s in samples)
        main_dims = dims_count.most_common(1)[0][0]
        dropped = len(samples) - sum(v for k, v in dims_count.items()
                                     if k == main_dims)
        samples = [s for s in samples if sample_dims(s) == main_dims]
        if dropped and verbose:
            print(f"  [filtro] descartadas {dropped} muestras fuera de "
                  f"dims {main_dims}")
    return samples, outcomes


async def amain(args):
    device = pick_device(args.device)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Rotar métricas al arrancar FRESCO: evita mezclar regímenes en el
    # dashboard (ya nos pasó — un trainer viejo vivo también puede seguir
    # agregando líneas huérfanas después del archivo rotado).
    if not args.resume and args.metrics and os.path.exists(args.metrics):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        os.replace(args.metrics, args.metrics + f".old_{stamp}")

    net = AlphaLiteNet()
    vocab = Vocab()
    trainer = PPOTrainer(net, lr=args.lr, device=device)

    start_iter = 0
    ckpt_extra: dict = {}
    if args.resume:
        # F2 (auditoría): restaurar TAMBIÉN el vocabulario del checkpoint —
        # antes se ignoraba y los ids de la cabeza de ítems podían quedar
        # barajados respecto a los pesos guardados.
        start_iter = load_checkpoint(
            args.resume, net,
            opt=None if args.reset_opt else trainer.opt,
            vocab=vocab, reset_opt=args.reset_opt,
            extra_out=ckpt_extra)
        print(f"Reanudado desde {args.resume} (iter {start_iter}, "
              f"{len(vocab.type_to_id)} tipos en vocab"
              f"{', Adam fresco' if args.reset_opt else ''})")

    bc_start_iter = int(getattr(args, "bc_start_iter", 0) or 0)
    if bc_start_iter <= 0:
        bc_start_iter = int(ckpt_extra.get("bc_start_iter") or 0)
    if args.bc and bc_start_iter <= 0:
        bc_start_iter = int(start_iter)

    if args.roles_vocab:
        # Traductor universal (agnóstico a facción): la cabeza de ítems pasa
        # a indexar ROLES estables (rl.roles), descartando los nombres
        # concretos del ckpt (que variaban por facción). Conserva los pesos
        # de la red; solo re-sembra el vocab de roles con ids deterministas.
        vocab.type_to_id = {}
        vocab.seed_roles()
        print(f"[roles-vocab] vocab de ROLES sembrado: "
              f"{len(vocab.type_to_id)} roles estables "
              f"(ids deterministas, agnóstico a facción)")

    infer_net = AlphaLiteNet().to(device)
    infer_net.load_state_dict(net.state_dict())
    infer_net.eval()

    print(f"Device: {device} | params: "
          f"{sum(p.numel() for p in net.parameters())/1e6:.2f}M")
    print("Capa 2: transformer 48 + scatter + cell|unidad (Net2Net)")
    print("Overlap: collect k+1 (infer_net) || update k (thread)")
    print(f"Server: {args.url} | {args.episodes} ep/iter, "
          f"k_skip={args.k_skip}, pool={args.concurrency}"
          + (f", MACRO {args.macro_ticks} t/decisión" if args.macro_ticks else ""))

    # Pool persistente de conexiones, repartido entre N servidores
    # (--url acepta lista separada por comas: un contenedor por URL)
    urls = [u.strip() for u in args.url.split(",") if u.strip()]
    pool_size = max(1, min(args.concurrency, args.episodes))
    pool = []
    # El timeout del WebSocket DEBE escalar con la duración del episodio.
    # Con max_steps fijo 160 fue el cuello: en partidas largas (max_steps alto
    # o macro_ticks real) un advance legítimo tarda mas que 160s y el cliente
    # cortaba con TimeoutError aunque el daemon siguiera trabajando
    # (fix 2026-08-25). Factor ~3s por decisión (cada una simula macro_ticks)
    # da margen real para episodios largos; --msg-timeout actua como piso.
    ws_timeout = max(args.msg_timeout, args.max_steps * 3.0)
    for i in range(pool_size):
        base_url = urls[i % len(urls)]
        pool.append(OpenRAEnv(base_url=base_url, message_timeout_s=ws_timeout))
    await asyncio.gather(*(env.connect() for env in pool))
    print(f"Pool: {pool_size} conexiones sobre {len(urls)} servidor(es) | "
          f"ws_timeout={ws_timeout:.0f}s")

    # Curriculum Fase 2: --scenario A resetea con el mapa pre-construido
    # (base del cliente + 8 rifles vs beginner). None = juego completo.
    reset_kwargs = {}
    if args.scenario:
        mapa = Path("rl/scenarios") / f"fase2_{args.scenario.lower()}.oramap"
        if not mapa.exists():
            raise SystemExit(f"escenario inexistente: {mapa} "
                             f"(generar con rl/make_scenario.py)")
        # map_name ESTABLE: el server sobreescribe siempre el mismo archivo
        # y el engine puede cachearlo. Sin esto genera un nombre unico por
        # episodio => cache fria en cada reset (~+100 s/iter medida).
        reset_kwargs = {"map_data": base64.b64encode(mapa.read_bytes()).decode(),
                        "map_name": f"fase2_{args.scenario.lower()}.oramap"}
        print(f"Escenario {args.scenario}: base pre-construida "
              f"({mapa.name}, b64 {len(reset_kwargs['map_data'])} chars)")

    # Oponente configurable por sesion: el server procesa bot_type en los
    # kwargs de reset (openra_environment.py). Curriculum de oponentes —
    # regla de casa: promover solo tras cumplir el criterio de winrate en
    # el nivel actual (un cambio de regimen a la vez).
    if args.bot_type:
        reset_kwargs["bot_type"] = args.bot_type
        print(f"Rival: bot_type={args.bot_type}")

    def launch_collection(prev_task):
        """Lanza una tanda de episodios repartidos entre los workers."""
        per_worker = [args.episodes // pool_size] * pool_size
        for i in range(args.episodes % pool_size):
            per_worker[i] += 1

        async def run_batch():
            results = []

            async def worker(idx, n):
                for _ in range(n):
                    traj, outcome = None, None
                    # Reset con reintentos: el daemon .NET agotado falla en
                    # reset ("bridge failed to start") aunque los healthchecks
                    # HTTP pasen. Reintenta y aborta limpio si es persistente.
                    # Además, DEADLINE_EXCEEDED del FastAdvance deja la sesión
                    # envenenada: el reset sobre el mismo /ws también puede
                    # colgar — en ese caso recrear la conexión WS.
                    for intento in range(2):
                        try:
                            traj, outcome = await collect_one_episode(
                                pool[idx], infer_net, vocab, device,
                                k_skip=args.k_skip,
                                temperature=args.temperature,
                                max_steps=args.max_steps,
                                macro_ticks=args.macro_ticks,
                                reset_kwargs=reset_kwargs,
                                shaper_preset=args.shaper_preset,
                                auto_support=args.auto_support)
                            break
                        except Exception as e:
                            msg = str(e)
                            is_deadline = "DEADLINE" in msg or "Deadline" in msg
                            is_bridge = ("bridge failed to start" in msg
                                         or "Session failed" in msg)
                            if is_deadline:
                                print(f"  [reset] DEADLINE en worker {idx} "
                                      f"intento {intento+1}/2 — recreando WS")
                                try:
                                    await pool[idx].close()
                                except Exception:
                                    pass
                                # Recrear la entrada del pool con nueva conexión
                                base_url = urls[idx % len(urls)]
                                pool[idx] = OpenRAEnv(base_url=base_url,
                                                      message_timeout_s=ws_timeout)
                                try:
                                    await pool[idx].connect()
                                except Exception as ce:
                                    print(f"  [reset] reconnect falló: {ce}")
                                    continue
                                # Reintentar el episodio con la nueva sesión
                                continue
                            if is_bridge:
                                print(f"  [reset] reintento {intento + 1}/2 "
                                      f"tras: {msg[:80]}")
                                continue
                            raise
                    if outcome is None:
                        print("  [reset] 2 fallos seguidos -> abortando run "
                              "(contenedores probablemente agotados)")
                        raise RuntimeError("reset persistente: recrear "
                                           "contenedores")
                    # Si el episodio abortó por DEADLINE (engine_error con
                    # outcome_error), la sesión pudo quedar envenenada aunque
                    # collect_one_episode ya intentó destroy. Forzar un reset
                    # de saneamiento best-effort antes del próximo episodio
                    # del mismo worker para no arrastrar el envenenamiento.
                    if outcome.get("result") == "engine_error":
                        try:
                            await asyncio.wait_for(
                                pool[idx].reset(**reset_kwargs), timeout=30)
                        except Exception:
                            # Si el saneamiento cuelga, recrear WS como arriba
                            try:
                                await pool[idx].close()
                            except Exception:
                                pass
                            base_url = urls[idx % len(urls)]
                            pool[idx] = OpenRAEnv(base_url=base_url,
                                                  message_timeout_s=ws_timeout)
                            try:
                                await pool[idx].connect()
                            except Exception:
                                pass
                    results.append((traj, outcome))

            await asyncio.gather(*(worker(i, n)
                                   for i, n in enumerate(per_worker)))
            return results

        # Esperamos la tanda anterior antes de reusar los envs del pool
        if prev_task is not None:
            pass  # el caller ya hizo await del task anterior
        return asyncio.ensure_future(run_batch())

    pending = None
    wins = total = 0
    # Los contadores de winrate son de la ERA, no del proceso: si el trainer
    # se reinicia (crash, deploy, cambio de régimen) se hidratan desde la
    # última fila del JSONL de métricas; la rotación de datos los resetea al
    # abrir cada era nueva. Antes, cada restart hacía retroceder el
    # denominador y el dashboard mostraba una era más corta que la real.
    # (winrate_rolling20 no se hidrata: se reconstruye con las próximas 20.)
    if os.path.exists(args.metrics):
        try:
            with open(args.metrics, encoding="utf-8") as _fmet:
                for _line in _fmet:
                    _line = _line.strip()
                    if not _line:
                        continue
                    try:
                        _jrow = json.loads(_line)
                    except json.JSONDecodeError:
                        continue  # marcadores {"note": ...} u otras líneas
                    if isinstance(_jrow.get("total"), int):
                        wins = int(_jrow.get("wins", 0) or 0)
                        total = _jrow["total"]
        except OSError as e:
            print(f"Aviso: no pude leer {args.metrics} ({e}); "
                  f"contadores arrancan en 0")
    if total:
        print(f"Contadores hidratados de la era: {wins}/{total} episodios")
    recent_results = []  # resultados recientes para winrate rodante (20)
    ema_collect = ema_update = None
    t_start = time.time()
    elite = EliteBuffer(cap_steps=2000) if args.sil else None
    if args.bc or args.sil:
        print(f"Capa 1: bc={args.bc} sil={args.sil} "
              f"warmup={args.bc_warmup} lambda_sil={args.lambda_sil} "
              f"bc_start_iter={bc_start_iter}", flush=True)

    for it in range(start_iter + 1, args.iters + 1):
        t0 = time.time()
        if pending is None:
            pending = launch_collection(None)
        results = await pending

        # Sync infer <- train (última update). Después arranca collect k+1
        # con esos pesos y el update k corre en un thread: el event loop
        # puede avanzar OpenRA. No tocar infer_net hasta el próximo await.
        infer_net.load_state_dict(net.state_dict())
        samples, outcomes = process_results(results, args.gamma, args.lam,
                                            adv_mode=args.adv_mode)
        lmb_bc = (lambda_bc_at(it, bc_start_iter, args.bc_warmup)
                  if args.bc else 0.0)
        lmb_sil = args.lambda_sil if args.sil else 0.0
        bc_samples = []
        bc_meta = {}
        if args.bc and lmb_bc > 0.0:
            try:
                t_traj, t_out = await collect_one_episode(
                    pool[0], infer_net, vocab, device,
                    k_skip=args.k_skip,
                    temperature=args.temperature,
                    max_steps=args.max_steps,
                    macro_ticks=args.macro_ticks,
                    reset_kwargs=reset_kwargs,
                    shaper_preset=args.shaper_preset,
                    auto_support=args.auto_support,
                    teacher=ScriptedTeacher())
                bc_samples, _ = process_results(
                    [(t_traj, t_out)], args.gamma, args.lam,
                    adv_mode=args.adv_mode, verbose=False)
                n_raw = len(bc_samples)
                bc_samples = balance_bc_samples(bc_samples)
                for s in bc_samples:
                    s["_ep"] = 10_000 + it
                bc_meta = {
                    "bc_n": len(bc_samples),
                    "bc_n_raw": n_raw,
                    "bc_result": t_out.get("result"),
                }
                print(f"  [bc] teacher steps={n_raw}->{len(bc_samples)} "
                      f"result={t_out.get('result')}", flush=True)
            except Exception as e:
                print(f"  [bc] teacher fail: {e}", flush=True)
                bc_samples = []
                bc_meta = {"bc_n": 0, "bc_result": "fail"}
        elif args.bc:
            print(f"  [bc] skip teacher (lambda_bc=0)", flush=True)
        pending = launch_collection(pending)

        t1 = time.time()
        skipped_update = batch_is_dead(outcomes)
        if elite is not None and samples:
            by_ep = {}
            for s in samples:
                by_ep.setdefault(s.get("_ep", 0), []).append(s)
            for ep_i, traj in by_ep.items():
                oc = outcomes[ep_i] if ep_i < len(outcomes) else {}
                elite.add_episode(traj, oc)
        if skipped_update:
            print(f"  [collapse] skip PPO update — batch >80% no_op "
                  f"(n={len(samples)})", flush=True)
            stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0,
                     "clip_frac": 0.0, "kl": 0.0, "grad_norm": 0.0,
                     "adv_mean": 0.0, "n": len(samples)}
            dt_update = 0.0
        else:
            sil_batch = (elite.sample_recent(512)
                         if (lmb_sil > 0.0 and elite is not None) else [])

            def _ppo_and_imitation():
                st = trainer.update(samples, args.epochs, args.batch_size)
                if lmb_bc > 0.0 and bc_samples:
                    st["bc_nll"] = trainer.imitation_update(
                        bc_samples, lmb_bc, epochs=1,
                        batch_size=args.batch_size)
                    st["lambda_bc"] = round(lmb_bc, 4)
                if lmb_sil > 0.0 and sil_batch:
                    st["sil_nll"] = trainer.imitation_update(
                        sil_batch, lmb_sil, epochs=1,
                        batch_size=args.batch_size)
                    st["sil_n"] = len(sil_batch)
                if device == "cuda":
                    torch.cuda.empty_cache()
                return st

            stats = await asyncio.to_thread(_ppo_and_imitation)
            dt_update = time.time() - t1
        dt = time.time() - t0  # wall-clock real (collect k + update, con overlap)
        collect_s = t1 - t0

        ema_collect = collect_s if ema_collect is None else \
            0.9 * ema_collect + 0.1 * collect_s
        ema_update = dt_update if ema_update is None else \
            0.9 * ema_update + 0.1 * dt_update
        eta_s = (ema_collect + ema_update * 0.3) * (args.iters - it)

        for o in outcomes:
            total += 1
            wins += 1 if str(o["result"]).startswith("win") else 0

        # Descomposición MEDIA del reward de esta tanda por componente:
        # combate / assets / buildings / new_types / margin. Si el reward
        # total se estanca, esto muestra QUÉ componente está plano.
        comp_means = {}
        if outcomes:
            for k in outcomes[0].get("reward_components", {}):
                comp_means[k] = round(sum(
                    o.get("reward_components", {}).get(k, 0.0)
                    for o in outcomes) / len(outcomes), 4)

        ckpt_path = os.path.join(args.ckpt_dir, "latest.pt")
        ckpt_blob = {"vocab": dict(vocab.type_to_id)}
        if args.bc:
            ckpt_blob["bc_start_iter"] = int(bc_start_iter)
        save_checkpoint(ckpt_path, net, trainer.opt, it, extra=ckpt_blob)
        if it % 10 == 0:
            save_checkpoint(os.path.join(args.ckpt_dir, f"iter{it:04d}.pt"),
                            net, trainer.opt, it, extra=ckpt_blob)

        if args.metrics:
            os.makedirs(os.path.dirname(args.metrics) or ".", exist_ok=True)
            elapsed_s = time.time() - t_start
            mean_ep_reward = (sum(o.get("episode_reward", 0.0)
                                  for o in outcomes) / len(outcomes)
                              if outcomes else 0.0)
            ticks_total = sum(o["ticks"] for o in outcomes)
            sim_tps = ticks_total / collect_s if collect_s > 0 else 0.0
            # winrate rodante: últimas 20 partidas (reacciona más rápido
            # que el global acumulado)
            recent_results.extend(o["result"] for o in outcomes)
            rolling = sum(1 for r in recent_results[-20:] if str(r).startswith("win")) \
                / min(len(recent_results), 20)
            # Modo macro: ticks avanzados vía advance() + interrupciones
            adv_total = sum(o.get("advanced_ticks", 0) for o in outcomes)
            int_count = {}
            for o in outcomes:
                for k, v in o.get("interrupts", {}).items():
                    int_count[k] = int_count.get(k, 0) + v
            # Supremacía media de la tanda + corpus por episodio (nivel 3).
            # Episodios con ambos bandos en $0 = sesión muerta del daemon:
            # se excluyen de medias y corpus (no son dominio nuestro).
            def _sup_valida(s):
                return bool(s) and bool(s.get("own") or s.get("enemy"))

            sups = [o.get("supremacy") for o in outcomes
                    if _sup_valida(o.get("supremacy"))]
            sup_mean = ({k: round(sum(s[k] for s in sups) / len(sups), 3)
                         for k in ("own", "enemy", "diff",
                                   "lead_ratio", "p_win_est")}
                        if sups else {})
            eps_corpus = [{"result": o.get("result"),
                           **(o.get("supremacy") or {})}
                          for o in outcomes
                          if _sup_valida(o.get("supremacy"))]
            # Carrera económica: agregados para métricas + series a archivo.
            # Mismo criterio: episodios sin nada propio son sesiones muertas.
            races = [o.get("economy_race") for o in outcomes
                     if o.get("economy_race")
                     and o["economy_race"].get("own_wealth_end", 0) > 0]
            race_mean = {}
            if races:
                race_mean = {
                    k: round(sum(r[k] for r in races) / len(races), 3)
                    for k in ("own_income_per_1k", "enemy_income_per_1k",
                              "income_edge", "own_harvest_per_1k",
                              "enemy_harvest_per_1k", "harvest_edge",
                              "peak_lead", "worst_deficit")
                }
            with open(args.race_file, "a", encoding="utf-8") as rf:
                for o in outcomes:
                    s = o.pop("economy_race_series", None)
                    if s and len(s.get("ticks", [])) >= 2:
                        rf.write(json.dumps({
                            "ts": datetime.now().isoformat(timespec="seconds"),
                            "iter": it,
                            "result": o.get("result"),
                            **s,
                        }) + "\n")
            # Nivel 2: resumen de las curvas V(s) del crítico por episodio
            vcs = [o["value_curve"] for o in outcomes if o.get("value_curve")]
            # Histograma de acciones efectivas de la tanda (sumado): ver qué
            # fracción es train/no_op vs el remate army_attack_move.
            hist_total = {}
            for o in outcomes:
                for k, v in o.get("action_hist", {}).items():
                    hist_total[k] = hist_total.get(k, 0) + v
            # Conteo espectador de edificios de cada bando al cierre
            # (promedio de la tanda; el rival baja cuando raseamos su base).
            nbs = [o["n_buildings"] for o in outcomes if o.get("n_buildings")]
            nb_mean = {}
            if nbs:
                nb_mean = {side: round(sum(b[side] for b in nbs) / len(nbs), 1)
                           for side in ("own", "enemy")}
            metrics_row = {
                    "iter": it,
                    "bot_type": args.bot_type,
                    "elapsed_s": round(elapsed_s, 1),
                    "eta_min": round(max(eta_s, 0) / 60, 1),
                    "collect_s": round(collect_s, 1),
                    "update_s": round(dt_update, 1),
                    "samples": stats.get("n", 0),
                    "pi_loss": stats.get("pi_loss"),
                    "v_loss": stats.get("v_loss"),
                    "entropy": stats.get("entropy"),
                    "clip_frac": stats.get("clip_frac"),
                    "kl": stats.get("kl"),
                    "grad_norm": stats.get("grad_norm"),
                    **({"lambda_bc": round(lmb_bc, 4)} if args.bc else {}),
                    **({"lambda_sil": round(lmb_sil, 4)} if args.sil else {}),
                    **bc_meta,
                    **({"bc_nll": stats.get("bc_nll")}
                       if stats.get("bc_nll") is not None else {}),
                    **({"sil_nll": stats.get("sil_nll")}
                       if stats.get("sil_nll") is not None else {}),
                    **({"sil_n": stats.get("sil_n")}
                       if stats.get("sil_n") is not None else {}),
                    **({"elite_n": len(elite)} if elite is not None else {}),
                    "winrate": round(wins / total, 3) if total else 0.0,
                    "winrate_rolling20": round(rolling, 3),
                    "iter_winrate": round(
                        (sum(1 for o in outcomes
                             if str(o.get("result", "")).startswith("win"))
                         / len(outcomes)) if outcomes else 0.0, 3),
                    "mean_episode_reward": round(mean_ep_reward, 4),
                    "reward_components": comp_means,
                    "sim_ticks_per_s": round(sim_tps),
                    "wins": wins,
                    "total": total,
                    "outcomes": [o["result"] for o in outcomes],
                    "ticks": [o["ticks"] for o in outcomes],
                    **({"advanced_ticks_per_ep": round(
                            adv_total / len(outcomes), 1)}
                       if outcomes and args.macro_ticks else {}),
                    **({"interrupts": int_count}
                       if args.macro_ticks else {}),
                    **({"supremacy": sup_mean} if sup_mean else {}),
                                        **({"sup_exact": bool(sups) and all(s.get("exact") for s in sups)}),
                                        **({"episodes_supremacy": eps_corpus}
                                            if sups else {}),
                    **({"economy_race": race_mean} if race_mean else {}),
                    **({"action_hist": hist_total} if hist_total else {}),
                    **({"n_buildings": nb_mean} if nb_mean else {}),
                    **({"critic_v_mean": round(
                            sum(sum(vc) for vc in vcs)
                            / sum(len(vc) for vc in vcs), 3)}
                       if vcs else {}),
                    **({"update_skipped": True} if skipped_update else {}),
            }
            with open(args.metrics, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics_row) + "\n")
            maybe_update_best(args.ckpt_dir, metrics_row, latest_path=ckpt_path)

        print(f"[iter {it:3d}] col {collect_s:5.1f}s upd {dt_update:5.1f}s "
              f"(ETA {eta_s/60:5.1f}m) | samples {stats.get('n', 0):4d} | "
              f"pi {stats['pi_loss']:+.4f} v {stats['v_loss']:.4f} "
              f"H {stats['entropy']:.3f} clip {stats['clip_frac']:.3f} "
              f"gn {stats['grad_norm']:.2f} | winrate {wins}/{total} | "
              f"c[cbt {comp_means.get('combat', 0):+.2f} "
              f"ast {comp_means.get('assets', 0):+.2f} "
              f"bld {comp_means.get('buildings', 0):+.2f} "
              f"typ {comp_means.get('new_types', 0):+.2f} "
              f"min {comp_means.get('mining', 0):+.2f} "
              f"mrg {comp_means.get('margin', 0):+.2f}] | "
              + (f"eco[cosecha nos {race_mean.get('own_harvest_per_1k', 0):+.0f} vs "
                 f"rival {race_mean.get('enemy_harvest_per_1k', 0):+.0f} | "
                 f"riqueza nos {race_mean.get('own_income_per_1k', 0):+.0f} vs "
                 f"rival {race_mean.get('enemy_income_per_1k', 0):+.0f} $/kt] | "
                 if race_mean else "")
              + " ".join(f"{o['result']}@{o['ticks']}t({o['wall_s']}s)"
                         for o in outcomes))

    if pending is not None:
        pending.cancel()
    await asyncio.gather(*(env.close() for env in pool),
                         return_exceptions=True)
    print(f"Listo. {total} episodios, winrate {wins}/{total}, "
          f"{time.time()-t_start:.0f}s totales.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=4,
                    help="partidas por iteración (grupo)")
    ap.add_argument("--k-skip", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=208,
                    help="env.steps por episodio (x2 ticks c/u; 6000≈8min juego). "
                         "En modo macro cuenta DECISIONES. Régimen 2-B: 208 "
                         "decisiones — sonda de horizonte midió que ninguna "
                         "declaración cabe en 104 (docs/sonda-horizonte.md)")
    ap.add_argument("--macro-ticks", type=int, default=0,
                    help=">0 activa modo v4-macro: presupuesto de ticks por "
                         "decisión vía advance() (ej. 160); 0 = frame-skip viejo")
    ap.add_argument("--concurrency", type=int, default=3,
                    help="sesiones simultáneas (el server soporta hasta 64)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ckpt-dir", default="rl/ckpts")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--metrics", default="rl/ckpts/metrics.jsonl",
                    help="JSONL de métricas por iteración (para dashboard)")
    ap.add_argument("--race-file", default="rl/ckpts/economy_race.jsonl",
                    help="series de carrera económica por episodio")
    ap.add_argument("--adv-mode", choices=("episode", "global", "none"),
                    default="episode",
                    help="normalizacion de ventajas (A/B revision externa: "
                         "'global' = Z-score del batch completo, conserva la "
                         "senal entre episodios; 'episode' = centrado grupal "
                         "historico)")
    ap.add_argument("--scenario", default=None,
                    help="Curriculum Fase 2: 'A'/'a_short' = base pre-construida "
                         "(rl/scenarios/fase2_{X}.oramap). Vacio = juego completo.")
    ap.add_argument("--bot-type", default=None,
                    choices=("beginner", "easy", "medium", "hard", "brutal",
                             "dummy"),
                    help="Personalidad del bot rival. 'dummy' = bot ESTATICO "
                         "(sin IA: no cosecha/construye/repara; base pre-"
                         "colocada no se regenera). Vacio = beginner.")
    ap.add_argument("--roles-vocab", action="store_true",
                    help="Traductor universal: sembrar la cabeza de items con "
                         "ROLES funcionales estables (rl.roles) en vez de "
                         "nombres concretos por facción. Al resume descarta el "
                         "vocab viejo del ckpt pero conserva los pesos de la red.")
    ap.add_argument("--msg-timeout", type=float, default=160.0,
                    help="Timeout (s) por DIALOGO agente<->motor. Los episodios "
                         "del daemon .NET a veces se cuelgan a mitad de un "
                         "advance(); con 600s quemaban ~11min de GPU esperando "
                         "el timeout. 160s recupera el throghput (a costa de "
                         "mas falsos engine_error si el daemon está lento).")
    ap.add_argument("--shaper-preset", choices=SHAPER_PRESETS,
                    default="eradicate",
                    help="Régimen de reward: 'eradicate' (combate asimétrico + "
                         "raze, objetivo Fase 2) o 'legacy' (SimCity histórico). "
                         "Un cambio de régimen por run.")
    ap.add_argument("--auto-support", action="store_true",
                    help="Pilar B: autonomía de soporte (repair hp<35%% + power_down) — "
                         "0 decisiones, gratis para PPO. Activo en Run3/v4.")
    ap.add_argument("--reset-opt", action="store_true",
                    help="Al --resume, no cargar Adam del ckpt (momentos de una "
                         "política colapsada clavan la cabeza de tipo). Lo pasa "
                         "auto_train tras restaurar best.pt.")
    ap.add_argument("--bc", action="store_true",
                    help="Capa 1: 1 episodio ScriptedTeacher por iter + NLL BC.")
    ap.add_argument("--sil", action="store_true",
                    help="Capa 1: self-imitation de episodios win/raze>0.")
    ap.add_argument("--bc-warmup", type=int, default=80,
                    help="Iters para bajar lambda_bc de 1.0 a 0.")
    ap.add_argument("--bc-start-iter", type=int, default=0,
                    help="Origen del warmup BC. 0 = ckpt o start_iter. "
                         "No debe resetearse en cada --resume.")
    ap.add_argument("--lambda-sil", type=float, default=0.5,
                    help="Peso SIL cuando --sil (default 0.5).")
    args = ap.parse_args()

    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\nInterrumpido — checkpoint latest.pt conserva la última iter.")


if __name__ == "__main__":
    main()
