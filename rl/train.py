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
from rl.best_ckpt import batch_is_dead, batch_is_wipe, maybe_update_best
from rl.pfsp import BotPFSP, parse_pool
from rl.imitation import (
    EliteBuffer, TeacherWinBuffer, SIL_PREFER_TICKS,
    BC_WIN_CAP, BC_WIN_PREFER_TICKS,
    balance_bc_samples, lambda_bc_at, merge_teacher_wins,
)
from rl.onboard import mix_target_prob
from rl.scripted_teacher import ScriptedTeacher
from rl import map_catalog as mapcat


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

    def _concat_advs():
        chunks = [
            np.array([s["adv"] for s in traj], dtype=np.float32)
            for traj in episodes if traj
        ]
        return np.concatenate(chunks) if chunks else None

    if adv_mode == "episode":
        center_advantage_by_episode(episodes)
        # F8: el escalado que antes hacía el trainer vive acá (centrado por
        # episodio + división por la std del BATCH COMPLETO ≈ escala histórica
        # exacta; ~1200 muestras, no grupos chicos).
        # 4 traj vacías (engine_error antes del primer step) → skip, no
        # np.concatenate([]) (auditoría 1.3).
        all_advs = _concat_advs()
        if all_advs is not None:
            sd = float(all_advs.std())
            if sd > 1e-8:
                for traj in episodes:
                    for s in traj:
                        s["adv"] = s["adv"] / sd
    elif adv_mode == "global":
        all_advs = _concat_advs()
        if all_advs is not None:
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
            # Safety net: shell map should already be skipped in rollout.
            print(f"  [filtro] descartadas {dropped} muestras fuera de "
                  f"dims {main_dims}")
    return samples, outcomes


async def collect_teacher_games(pool, net, vocab, device, args, reset_kwargs):
    """N partidas ScriptedTeacher en paralelo sobre el pool de envs."""
    n = max(1, int(getattr(args, "bc_games", 1) or 1))
    t_kwargs = dict(reset_kwargs or {})
    teacher_bot = getattr(args, "bc_teacher_bot", None)
    if teacher_bot:
        t_kwargs["bot_type"] = teacher_bot
    envs = list(pool or [])
    if not envs:
        raise RuntimeError("collect_teacher_games: pool vacío")

    async def _one(i, env):
        t_macro = int(getattr(args, "bc_macro_ticks", 0) or 0) or args.macro_ticks
        t_steps = int(getattr(args, "bc_max_steps", 0) or 0) or args.max_steps
        ep_kwargs = dict(t_kwargs)
        pool = mapcat.parse_pool_arg(getattr(args, "map_pool", None))
        if pool:
            ep_kwargs.update(mapcat.reset_payload_for(mapcat.sample_pool(pool)))
        try:
            traj, outcome = await collect_one_episode(
                env, net, vocab, device,
                k_skip=args.k_skip,
                temperature=args.temperature,
                max_steps=t_steps,
                macro_ticks=t_macro,
                reset_kwargs=ep_kwargs,
                shaper_preset=args.shaper_preset,
                auto_support=args.auto_support,
                war_nudge=not args.no_war_nudge,
                teacher=ScriptedTeacher(
                    rush_attack_move=int(getattr(args, "bc_rush", 0) or 0) or None))
        except Exception as e:
            print(f"  [bc] teacher game {i + 1}/{n} fail: {e}", flush=True)
            return None
        samples, _ = process_results(
            [(traj, outcome)], args.gamma, args.lam,
            adv_mode=args.adv_mode, verbose=False)
        print(f"  [bc] teacher {i + 1}/{n} "
              f"result={outcome.get('result')} ticks={outcome.get('ticks')}",
              flush=True)
        return samples, outcome

    episodes = []
    for start in range(0, n, len(envs)):
        chunk = list(range(start, min(start + len(envs), n)))
        parts = await asyncio.gather(*[
            _one(i, envs[j % len(envs)]) for j, i in enumerate(chunk)
        ])
        for p in parts:
            if p is not None:
                episodes.append(p)
    # Wins-only by default. Incomplete tapes teach timeout-turtle; only keep
    # them with an explicit --bc-keep-incomplete (legacy opening clone).
    # Return per-episode dicts for TeacherWinBuffer; balance happens after
    # sampling the persistent ring in amain.
    keep_inc = bool(getattr(args, "bc_keep_incomplete", False))
    kept, meta, kept_eps = merge_teacher_wins(
        episodes,
        keep_incomplete=keep_inc)
    meta["bc_n"] = len(kept)
    meta["bc_n_raw"] = len(kept)
    if not kept_eps:
        print(f"  [bc] 0 samples (wins={meta.get('bc_n_win_eps', 0)} "
              f"inc={meta.get('bc_n_incomplete_eps', 0)}/"
              f"{meta.get('bc_n_eps', 0)}; no clonar derrotas)", flush=True)
    else:
        print(f"  [bc] keep wins={meta.get('bc_n_win_eps', 0)} "
              f"inc={meta.get('bc_n_incomplete_eps', 0)}/"
              f"{meta.get('bc_n_eps', 0)} steps={len(kept)} "
              f"eps={len(kept_eps)}",
              flush=True)
    return kept_eps, meta


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
    if getattr(args, "xf_topk", 0):
        net.xf_topk = int(args.xf_topk)
        print(f"entity XF top-k={net.xf_topk}", flush=True)
    if getattr(args, "qsa_topk", 0):
        net.qsa_topk = int(args.qsa_topk)
        net.qsa_block = int(getattr(args, "qsa_block", 8) or 8)
        print(f"map QSA top-k={net.qsa_topk} block={net.qsa_block}", flush=True)
    vocab = Vocab()
    trainer = PPOTrainer(
        net, lr=args.lr, device=device,
        clip_eps=args.clip_eps, max_grad_norm=args.max_grad_norm,
        burn_in_len=args.burn_in,
        amp_init_scale=float(getattr(args, "amp_init_scale", 0) or 0) or None,
        use_amp=False if getattr(args, "no_amp", False) else None,
    )

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
        print(f"[roles-vocab] {len(vocab.type_to_id)} roles", flush=True)


    infer_net = AlphaLiteNet().to(device)
    infer_net.load_state_dict(net.state_dict())
    infer_net.eval()

    def load_opponent_net(ckpt_path):
        """Frozen AlphaLite for Multi0 (RL-vs-RL). None if load fails."""
        if not ckpt_path:
            return None
        try:
            opp = AlphaLiteNet().to(device)
            load_checkpoint(str(ckpt_path), opp, opt=None, vocab=None)
            opp.eval()
            for p in opp.parameters():
                p.requires_grad_(False)
            return opp
        except Exception as e:
            print(f"  [rvr] no pude cargar oponente {ckpt_path}: {e}", flush=True)
            return None

    _bc_only = bool(getattr(args, "bc_only", False))
    _bc_collect_only = bool(getattr(args, "bc_collect_only", False))
    if _bc_collect_only:
        args.bc = True
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    mode = []
    if _bc_collect_only:
        mode.append("BC-COLLECT-ONLY")
    elif _bc_only:
        mode.append("BC-ONLY/SFT")
    else:
        mode.append("PPO")
        if getattr(args, "bc", False):
            mode.append("BC")
        if getattr(args, "sil", False):
            mode.append("SIL")
    if getattr(args, "onboard_phase", None):
        mode.append(f"onboard={args.onboard_phase}")
    print(
        f"Device: {device} | params: {n_params:.2f}M | "
        f"{'+'.join(mode)} | server={args.url} | "
        f"ep={args.episodes} k_skip={args.k_skip} pool={args.concurrency}"
        + (f" macro={args.macro_ticks}" if args.macro_ticks else ""),
        flush=True,
    )

    # Pool persistente de conexiones, repartido entre N servidores
    # (--url acepta lista separada por comas: un contenedor por URL)
    urls = [u.strip() for u in args.url.split(",") if u.strip()]
    teach_n = max(1, int(getattr(args, "bc_games", 1) or 1)) if (
        getattr(args, "bc", False) or getattr(args, "bc_only", False)
    ) else 0
    pool_size = max(1, min(args.concurrency, max(args.episodes, teach_n)))
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
    print(f"Pool: {pool_size} envs / {len(urls)} urls | ws_timeout={ws_timeout:.0f}s",
          flush=True)

    # Maps: --scenario (single) and/or --map-pool (sample per episode).
    # Default curriculum stays a_short via auto_train; official mixed maps
    # opt-in via --map-pool official_2p_small|mixed|water.
    reset_kwargs = {}
    map_pool = mapcat.parse_pool_arg(getattr(args, "map_pool", None))
    if map_pool:
        print(f"Map pool ({len(map_pool)}): {','.join(map_pool)}", flush=True)
        # Seed reset_kwargs with first entry so teacher BC / early path has a map.
        try:
            reset_kwargs.update(mapcat.reset_payload_for(map_pool[0]))
            print(f"Map pool seed: {map_pool[0]} ({reset_kwargs.get('map_name')})",
                  flush=True)
        except (KeyError, FileNotFoundError) as e:
            raise SystemExit(f"map pool invalid: {e}") from e
    elif args.scenario:
        try:
            reset_kwargs.update(mapcat.reset_payload_for(args.scenario))
            entry = mapcat.get_entry(args.scenario)
            print(
                f"Escenario {entry.key} ({entry.file_name}) "
                f"water={entry.has_water}",
                flush=True)
        except KeyError:
            # Legacy fallback: fase2_{name}.oramap if not in catalog
            mapa = Path("rl/scenarios") / f"fase2_{args.scenario.lower()}.oramap"
            if not mapa.exists():
                raise SystemExit(
                    f"escenario inexistente: {args.scenario!r} / {mapa} "
                    f"(known: {', '.join(sorted(mapcat.MAP_CATALOG))})")
            reset_kwargs = {
                "map_data": base64.b64encode(mapa.read_bytes()).decode(),
                "map_name": mapa.name,
            }
            print(f"Escenario legacy {args.scenario} ({mapa.name})", flush=True)
        except FileNotFoundError as e:
            raise SystemExit(str(e)) from e

    # Oponente configurable por sesion: el server procesa bot_type en los
    # kwargs de reset (openra_environment.py). Curriculum de oponentes —
    # regla de casa: promover solo tras cumplir el criterio de winrate en
    # el nivel actual (un cambio de regimen a la vez).
    if args.bot_type:
        reset_kwargs["bot_type"] = args.bot_type
        print(f"Rival: {args.bot_type}", flush=True)

    pfsp = None
    if getattr(args, "pfsp", False):
        anchor = args.bot_type or "easy"
        pfsp = BotPFSP(
            args.ckpt_dir,
            anchor=anchor,
            pool=parse_pool(getattr(args, "pfsp_pool", None)),
            anchor_prob=float(getattr(args, "pfsp_anchor_prob", 0.5) or 0.5),
            prev20_every=int(getattr(args, "pfsp_prev20_every", 20) or 20),
        )
        print(
            f"PFSP bots ON: {int(pfsp.anchor_prob*100)}% vs {pfsp.anchor}, "
            f"else challengers={pfsp.challengers} (priority=who beats you). "
            f"North-star wr / best.pt solo cuentan vs {pfsp.anchor}.",
            flush=True,
        )
        if getattr(args, "pfsp_rl", False):
            if "rl" not in pfsp.pool:
                pfsp.pool.append("rl")
                pfsp.stats.setdefault("rl", {"wins": 0, "games": 0})
            print("PFSP-RL ON: pool puede samplear bot_type=rl (frozen Multi0).", flush=True)

    mix_from = getattr(args, "mix_from", None) or None
    mix_warmup = int(getattr(args, "mix_warmup", 0) or 0)
    mix_start_p = float(getattr(args, "mix_start", 0.25) or 0.25)
    mix_start_iter = int(getattr(args, "mix_start_iter", 0) or 0)
    mixing = bool(
        mix_from and args.bot_type and mix_from != args.bot_type
        and pfsp is None)
    collect_it = [0]
    if mixing:
        print(
            f"Mix ON: P({args.bot_type}) ramps {mix_start_p:.2f}→1.0 "
            f"over {mix_warmup} iters from {mix_from} "
            f"(start_iter={mix_start_iter or 'first'}). "
            f"North-star wr / best.pt solo vs {args.bot_type}.",
            flush=True,
        )

    if args.auto_support:
        if args.no_war_nudge:
            print("auto-support=on war_nudge=off", flush=True)
        else:
            print("auto-support=on war_nudge=on", flush=True)

    bc_only = bool(getattr(args, "bc_only", False))
    if bc_only:
        args.bc = True
        pass  # mode already in banner
    if getattr(args, "onboard_phase", None):
        pass  # in banner

    def launch_collection(prev_task):
        """Lanza una tanda de episodios repartidos entre los workers."""
        per_worker = [args.episodes // pool_size] * pool_size
        for i in range(args.episodes % pool_size):
            per_worker[i] += 1

        async def run_batch():
            results = []
            it_now = int(collect_it[0] or mix_start_iter or 1)
            p_tgt = 1.0
            if mixing:
                start_it = mix_start_iter or it_now
                p_tgt = mix_target_prob(
                    it_now, start_it, mix_warmup, mix_start_p)
                print(f"  [mix] it={it_now} p({args.bot_type})={p_tgt:.2f} "
                      f"from {mix_from}", flush=True)

            async def worker(idx, n):
                for _ in range(n):
                    traj, outcome = None, None
                    ep_kwargs = dict(reset_kwargs)
                    if map_pool:
                        mk = mapcat.sample_pool(map_pool)
                        ep_kwargs.update(mapcat.reset_payload_for(mk))
                    ep_bot = ep_kwargs.get("bot_type") or args.bot_type or "easy"
                    if pfsp is not None:
                        ep_bot = pfsp.sample()
                        ep_kwargs["bot_type"] = ep_bot
                    elif mixing:
                        start_it = mix_start_iter or it_now
                        p_now = mix_target_prob(
                            it_now, start_it, mix_warmup, mix_start_p)
                        if float(np.random.random()) >= p_now:
                            ep_bot = mix_from
                            ep_kwargs["bot_type"] = ep_bot
                    opp_net = None
                    if ep_bot == "rl":
                        ckpt = pfsp.pick_rl_ckpt() if pfsp is not None else None
                        opp_net = load_opponent_net(ckpt)
                        if opp_net is None:
                            # Fallback: no dual weights → easy scripted
                            ep_bot = (pfsp.anchor if pfsp is not None else "easy")
                            ep_kwargs["bot_type"] = ep_bot
                        else:
                            ep_kwargs["bot_type"] = "rl"
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
                                reset_kwargs=ep_kwargs,
                                shaper_preset=args.shaper_preset,
                                auto_support=args.auto_support,
                                war_nudge=not args.no_war_nudge,
                                opponent_net=opp_net)
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
                    if outcome is not None:
                        outcome["bot_type"] = ep_bot
                    if outcome.get("result") == "engine_error":
                        try:
                            await asyncio.wait_for(
                                pool[idx].reset(**ep_kwargs), timeout=30)
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
    elite_path = os.path.join(args.ckpt_dir, "elite.pt")
    if elite is not None:
        n_elite = elite.load(elite_path)
        if n_elite:
            print(f"[sil] loaded elite steps={n_elite} from {elite_path}",
                  flush=True)
    teacher_wins = None
    if args.bc or bc_only:
        win_cap = int(getattr(args, "bc_win_cap", 0) or BC_WIN_CAP)
        win_prefer = int(getattr(args, "bc_win_prefer_ticks", 0)
                         or BC_WIN_PREFER_TICKS)
        win_dir = getattr(args, "bc_win_dir", None) or os.path.join(
            args.ckpt_dir, "teacher_wins")
        teacher_wins = TeacherWinBuffer(
            cap_steps=win_cap,
            prefer_ticks=win_prefer,
            path=win_dir,
            keep_incomplete=bool(getattr(args, "bc_keep_incomplete", False)),
        )
        print(f"  [bc] TeacherWinBuffer cap={win_cap} prefer_ticks={win_prefer} "
              f"dir={win_dir} loaded eps={teacher_wins.n_episodes} "
              f"steps={len(teacher_wins)}",
              flush=True)

    # --bc-collect-only: accumulate teacher_wins then exit (no SFT/eval/ckpt).
    # auto_train --onboard-collect-only passes this; keeps latest.pt untouched.
    if bool(getattr(args, "bc_collect_only", False)):
        if teacher_wins is None:
            raise SystemExit("--bc-collect-only necesita --bc / --bc-only")
        target = max(1, int(getattr(args, "bc_collect_target", 40) or 40))
        print(
            f"[bc-collect-only] target={target} wins={teacher_wins.n_episodes} "
            f"steps={len(teacher_wins)} (no SFT, no eval, no ckpt write)",
            flush=True,
        )

        def _heartbeat(new_wins: int = 0) -> None:
            if not args.metrics:
                return
            os.makedirs(os.path.dirname(args.metrics) or ".", exist_ok=True)
            row = {
                "note": "bc_collect_only",
                "bc_collect_only": True,
                "bc_buffer_eps": teacher_wins.n_episodes,
                "bc_buffer_steps": len(teacher_wins),
                "bc_collect_target": target,
                "bc_new_wins": int(new_wins),
            }
            with open(args.metrics, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

        if teacher_wins.n_episodes >= target:
            print(
                f"[bc-collect-only] DONE already "
                f"{teacher_wins.n_episodes}/{target} wins "
                f"steps={len(teacher_wins)} — exit 0",
                flush=True,
            )
            _heartbeat(0)
            await asyncio.gather(*(env.close() for env in pool),
                                 return_exceptions=True)
            return

        round_i = 0
        while teacher_wins.n_episodes < target:
            round_i += 1
            need = target - teacher_wins.n_episodes
            print(
                f"[bc-collect-only] round {round_i} "
                f"wins={teacher_wins.n_episodes}/{target} "
                f"steps={len(teacher_wins)} need={need}",
                flush=True,
            )
            try:
                new_eps, bc_meta = await collect_teacher_games(
                    pool, infer_net, vocab, device, args, reset_kwargs)
            except Exception as e:
                print(f"  [bc-collect-only] teacher fail: {e}", flush=True)
                _heartbeat(0)
                continue
            new_wins = 0
            for ep in new_eps or []:
                n_add = teacher_wins.add_episode(
                    ep.get("steps") or [],
                    {"result": ep.get("result"), "ticks": ep.get("ticks")})
                if n_add > 0:
                    new_wins += 1
            if new_wins:
                teacher_wins.save()
            print(
                f"[bc-collect-only] +new_wins={new_wins} "
                f"wins={teacher_wins.n_episodes}/{target} "
                f"steps={len(teacher_wins)} "
                f"meta_wins={bc_meta.get('bc_n_win_eps', 0)}",
                flush=True,
            )
            _heartbeat(new_wins)

        print(
            f"[bc-collect-only] DONE {teacher_wins.n_episodes}/{target} wins "
            f"steps={len(teacher_wins)} — exit 0",
            flush=True,
        )
        await asyncio.gather(*(env.close() for env in pool),
                             return_exceptions=True)
        return

    if args.bc or args.sil:
        print(
            f"BC games={getattr(args, 'bc_games', 1)} epochs={getattr(args, 'bc_epochs', 1)} "
            f"teacher={getattr(args, 'bc_teacher_bot', None)} "
            f"start_iter={bc_start_iter}",
            flush=True,
        )

    # --iters = ultima iter INCLUSIVE (absoluto). Scratch --iters 100 => 1..100.
    # El +1 del range es el stop exclusivo de Python, no una iter extra.
    first_it = int(start_iter) + 1
    last_it = int(args.iters)
    if first_it > last_it:
        print(f"Nada que entrenar: start_iter={start_iter} >= --iters {last_it}",
              flush=True)
        return
    n_updates = last_it - first_it + 1
    print(f"Entrenando iters {first_it}..{last_it} inclusive ({n_updates} updates)",
          flush=True)
    bc_epochs = max(1, int(getattr(args, "bc_epochs", 1) or 1))
    collect_it[0] = first_it
    for it in range(first_it, last_it + 1):
        collect_it[0] = it
        t0 = time.time()
        if bc_only:
            infer_net.load_state_dict(net.state_dict())
            results = []
            samples, outcomes = [], []
            eval_n = int(getattr(args, "eval_games", 0) or 0)
            if eval_n > 0:
                saved_ep = int(args.episodes)
                saved_temp = float(args.temperature)
                args.episodes = eval_n
                # Phase A / bc_only: greedy eval (temp=1.0 samples junk buildings).
                args.temperature = 1.0
                try:
                    results = await launch_collection(None)
                    samples, outcomes = process_results(
                        results, args.gamma, args.lam,
                        adv_mode=args.adv_mode)
                    print(f"  [eval] student n={len(outcomes)} temp=0.0 "
                          f"{[o.get('result') for o in outcomes]}",
                          flush=True)
                finally:
                    args.episodes = saved_ep
                    args.temperature = saved_temp
        else:
            if pending is None:
                pending = launch_collection(None)
            results = await pending
            # Sync infer <- train (última update). Después arranca collect k+1
            # con esos pesos y el update k corre en un thread: el event loop
            # puede avanzar OpenRA. No tocar infer_net hasta el próximo await.
            infer_net.load_state_dict(net.state_dict())
            samples, outcomes = process_results(results, args.gamma, args.lam,
                                                adv_mode=args.adv_mode)
        lmb_bc = (1.0 if bc_only else
                  (lambda_bc_at(it, bc_start_iter, args.bc_warmup,
                                end=float(getattr(args, "bc_lambda_end", 0.0) or 0.0))
                   if args.bc else 0.0))
        lmb_sil = args.lambda_sil if args.sil else 0.0
        bc_samples = []
        bc_meta = {}
        if args.bc and lmb_bc > 0.0:
            try:
                replay = (
                    bool(getattr(args, "bc_replay", False))
                    and teacher_wins is not None
                    and teacher_wins.n_episodes > 0
                )
                if replay:
                    raw = teacher_wins.sample(max_steps=teacher_wins.cap)
                    bc_samples = balance_bc_samples(raw)
                    bc_meta = {
                        "bc_n": len(bc_samples),
                        "bc_n_raw": len(raw),
                        "bc_buffer_eps": teacher_wins.n_episodes,
                        "bc_buffer_steps": len(teacher_wins),
                        "bc_new_wins": 0,
                        "bc_replay": True,
                        "bc_n_win_eps": 0,
                        "bc_n_eps": 0,
                        "bc_results": [],
                    }
                    print(f"  [bc] replay tapes eps={teacher_wins.n_episodes} "
                          f"steps={len(teacher_wins)} sample={len(bc_samples)} "
                          f"(no collect)",
                          flush=True)
                else:
                    new_eps, bc_meta = await collect_teacher_games(
                        pool, infer_net, vocab, device, args, reset_kwargs)
                    new_wins = 0
                    if teacher_wins is not None:
                        for ep in new_eps or []:
                            n_add = teacher_wins.add_episode(
                                ep.get("steps") or [],
                                {"result": ep.get("result"),
                                 "ticks": ep.get("ticks")})
                            if n_add > 0:
                                new_wins += 1
                        if new_wins:
                            teacher_wins.save()
                        raw = teacher_wins.sample(max_steps=teacher_wins.cap)
                        bc_samples = balance_bc_samples(raw)
                        print(f"  [bc] buffer eps={teacher_wins.n_episodes} "
                              f"steps={len(teacher_wins)} (+new_wins={new_wins}) "
                              f"sample={len(bc_samples)}", flush=True)
                        bc_meta["bc_n"] = len(bc_samples)
                        bc_meta["bc_buffer_eps"] = teacher_wins.n_episodes
                        bc_meta["bc_buffer_steps"] = len(teacher_wins)
                        bc_meta["bc_new_wins"] = new_wins
                    else:
                        # Fallback: flatten this-iter wins (no persistent buffer).
                        flat = []
                        for ep in new_eps or []:
                            flat.extend(ep.get("steps") or [])
                        bc_samples = balance_bc_samples(flat)
                        bc_meta["bc_n"] = len(bc_samples)
                for s in bc_samples:
                    s["_ep"] = 10_000 + it
            except Exception as e:
                print(f"  [bc] teacher fail: {e}", flush=True)
                bc_meta = {"bc_n": 0, "bc_result": "fail"}
                if teacher_wins is not None and len(teacher_wins) > 0:
                    bc_samples = balance_bc_samples(
                        teacher_wins.sample(max_steps=teacher_wins.cap))
                    for s in bc_samples:
                        s["_ep"] = 10_000 + it
                    bc_meta["bc_n"] = len(bc_samples)
                    print(f"  [bc] buffer fallback sample={len(bc_samples)}",
                          flush=True)
                else:
                    bc_samples = []
        elif args.bc:
            print(f"  [bc] skip teacher (lambda_bc=0)", flush=True)
        if not bc_only:
            collect_it[0] = it + 1
            pending = launch_collection(pending)

        t1 = time.time()
        wiped = ((not bc_only)
                 and getattr(args, "onboard_phase", None) == "B"
                 and batch_is_wipe(outcomes))
        skipped_update = (not bc_only) and (batch_is_dead(outcomes) or wiped)
        if skipped_update:
            why = ("wipe all-lose <15k" if wiped else "batch >80% no_op")
            print(f"  [collapse] skip PPO update — {why} "
                  f"(n={len(samples)}); BC/SIL siguen", flush=True)
        if elite is not None and samples:
            by_ep = {}
            for s in samples:
                by_ep.setdefault(s.get("_ep", 0), []).append(s)
            for ep_i, traj in by_ep.items():
                oc = outcomes[ep_i] if ep_i < len(outcomes) else {}
                elite.add_episode(traj, oc)
            if len(elite) > 0:
                try:
                    elite.save(elite_path)
                except OSError as e:
                    print(f"[sil] save elite fail: {e}", flush=True)
        if bc_only:
            def _imitation_only():
                st = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0,
                      "clip_frac": 0.0, "kl": 0.0, "grad_norm": 0.0,
                      "adv_mean": 0.0, "n": len(bc_samples)}
                if lmb_bc > 0.0 and bc_samples:
                    st["bc_nll"] = trainer.imitation_update(
                        bc_samples, lmb_bc, epochs=bc_epochs,
                        batch_size=args.batch_size)
                    st["lambda_bc"] = round(lmb_bc, 4)
                if device == "cuda":
                    torch.cuda.empty_cache()
                return st

            stats = await asyncio.to_thread(_imitation_only)
            dt_update = time.time() - t1
        else:
            # Even-pick per win, prefer ticks<40k (not the tail of 1–2 longs).
            sil_batch = (elite.sample_recent(512)
                         if (lmb_sil > 0.0 and elite is not None) else [])

            def _ppo_and_imitation():
                if skipped_update:
                    st = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0,
                          "clip_frac": 0.0, "kl": 0.0, "grad_norm": 0.0,
                          "adv_mean": 0.0, "n": len(samples)}
                else:
                    st = trainer.update(samples, args.epochs, args.batch_size)
                if lmb_bc > 0.0 and bc_samples:
                    st["bc_nll"] = trainer.imitation_update(
                        bc_samples, lmb_bc, epochs=bc_epochs,
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
        eta_s = (ema_collect + ema_update * 0.3) * max(0, last_it - it)

        if pfsp is not None:
            pfsp.record_many(outcomes)
        wr_anchor = None
        if pfsp is not None:
            wr_anchor = pfsp.anchor
        elif mixing:
            wr_anchor = args.bot_type
        for o in outcomes:
            if wr_anchor and o.get("bot_type") != wr_anchor:
                continue
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
        if pfsp is not None and pfsp.maybe_rotate_prev20(it, ckpt_path):
            print(f"  [pfsp] prev20.pt <- latest @ iter {it}", flush=True)

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
            if wr_anchor:
                recent_results.extend(
                    o["result"] for o in outcomes
                    if o.get("bot_type") == wr_anchor)
            else:
                recent_results.extend(o["result"] for o in outcomes)
            rolling = (sum(1 for r in recent_results[-20:] if str(r).startswith("win"))
                       / min(len(recent_results), 20)) if recent_results else 0.0
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
            if wr_anchor:
                anchor_outs = [o for o in outcomes if o.get("bot_type") == wr_anchor]
            else:
                anchor_outs = list(outcomes)
            mix_p = None
            if mixing:
                start_it = mix_start_iter or it
                mix_p = mix_target_prob(it, start_it, mix_warmup, mix_start_p)
            metrics_row = {
                    "iter": it,
                    "bot_type": (pfsp.anchor if pfsp is not None else args.bot_type),
                    **({"onboard_phase": args.onboard_phase}
                       if getattr(args, "onboard_phase", None) else {}),
                    **({"bc_only": True} if bc_only else {}),
                    **({"pfsp": True,
                        "pfsp_pool": list(pfsp.pool),
                        "pfsp_stats": pfsp.summary(),
                        "opponent_bots": [o.get("bot_type") for o in outcomes]}
                       if pfsp is not None else {}),
                    **({"mix": True,
                        "mix_from": mix_from,
                        "mix_p_target": round(mix_p, 3),
                        "opponent_bots": [o.get("bot_type") for o in outcomes]}
                       if mixing else {}),
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
                    **({"amp_skip_frac": stats.get("amp_skip_frac"),
                        "amp_scale": stats.get("amp_scale"),
                        "amp_enabled": stats.get("amp_enabled")}
                       if stats.get("amp_skip_frac") is not None else {}),
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
                        (sum(1 for o in anchor_outs
                             if str(o.get("result", "")).startswith("win"))
                         / len(anchor_outs))
                        if anchor_outs else 0.0, 3),
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
    ap.add_argument("--iters", type=int, default=100,
                    help="Ultima iter inclusive (absoluto). Scratch: --iters 100 corre 1..100. Resume desde 1141: --iters 1161 corre 20 mas.")
    ap.add_argument("--episodes", type=int, default=4,
                    help="partidas por iteración (grupo)")
    ap.add_argument("--k-skip", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=208,
                    help="env.steps por episodio (x2 ticks c/u; 6000≈8min juego). "
                         "En modo macro cuenta DECISIONES. Régimen 2-B: 208 "
                         "decisiones — sonda de horizonte midió que ninguna "
                         "declaración cabe en 104 (rl/docs/_archive/eras/sonda-horizonte.md)")
    ap.add_argument("--macro-ticks", type=int, default=0,
                    help=">0 activa modo v4-macro: presupuesto de ticks por "
                         "decisión vía advance() (ej. 160); 0 = frame-skip viejo")
    ap.add_argument("--concurrency", type=int, default=3,
                    help="sesiones simultáneas (el server soporta hasta 64)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip-eps", type=float, default=0.2,
                    help="PPO ratio clip epsilon (Informe-3 Run45: 0.15)")
    ap.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="grad clip norm (Informe-3 Run45: 1.0)")
    ap.add_argument("--burn-in", type=int, default=0,
                    help="GRU burn-in steps before each BPTT segment (R2D2; Run46: 8)")
    ap.add_argument("--xf-topk", type=int, default=0,
                    help="entity transformer top-k attention (0=dense softmax; Run46: 16)")
    ap.add_argument("--qsa-topk", type=int, default=0,
                    help="map QSA: keep top-k spatial blocks for cell head (0=off; Run47: 8)")
    ap.add_argument("--qsa-block", type=int, default=8,
                    help="map QSA block size in cells (default 8)")
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
                    help="Map/scenario key (catalog): a_short (default; lakes, "
                         "navy gated), doughnut / bombardment_islands / "
                         "tournament_island / x_lake (official mixed land-water), "
                         "or legacy fase2_{name}. Vacio = juego completo (singles).")
    ap.add_argument("--map-pool", default=None,
                    help="P3: sample a map per episode. Named pools: "
                         "land|mixed|official_2p_small|water "
                         "(water = naval-viable mixed, not pure-water), "
                         "or comma keys (e.g. a_short,doughnut). Overrides "
                         "single --scenario for collection; auto_train default "
                         "stays a_short when unset.")
    ap.add_argument("--bot-type", default=None,
                    choices=("beginner", "easy", "medium", "hard", "brutal",
                             "dummy"),
                    help="Personalidad del bot rival. Con --pfsp es el ANCLA "
                         "(north star + fraction of games).")
    ap.add_argument("--pfsp", action="store_true",
                    help="PFSP pobre de bots: fraction vs --bot-type (ancla), "
                         "resto vs pool priorizado al que mas te gana. "
                         "No es RL-vs-RL.")
    ap.add_argument("--pfsp-pool", default="beginner,easy,medium",
                    help="Bots del pool PFSP, separados por coma.")
    ap.add_argument("--pfsp-anchor-prob", type=float, default=0.5,
                    help="Probabilidad de jugar vs el ancla (--bot-type).")
    ap.add_argument("--pfsp-rl", action="store_true",
                    help="Incluye oponente RL (bot_type=rl) en PFSP; requiere daemon dual.")
    ap.add_argument("--pfsp-prev20-every", type=int, default=20,
                    help="Cada N iters copia latest.pt -> prev20.pt.")
    ap.add_argument("--mix-from", default=None,
                    choices=("beginner", "easy", "medium", "hard", "brutal",
                             "dummy"),
                    help="Con --bot-type: rampa P(bot-type) mezclando este rival "
                         "más fácil. wr20 / best.pt solo vs --bot-type. "
                         "No usa PFSP. Fase C: beginner→easy.")
    ap.add_argument("--mix-warmup", type=int, default=40,
                    help="Iters para subir P(--bot-type) de --mix-start a 1.0.")
    ap.add_argument("--mix-start", type=float, default=0.25,
                    help="P(--bot-type) al --mix-start-iter (default 0.25).")
    ap.add_argument("--mix-start-iter", type=int, default=0,
                    help="Iter donde P=mix-start (0 = primer iter de este run).")
    ap.add_argument("--amp-init-scale", type=float, default=0.0,
                    help="GradScaler init. 0 = default PyTorch (65536).")
    ap.add_argument("--no-amp", action="store_true",
                    help="Disable CUDA AMP (fp32). Phase C must not use AMP: "
                         "skip_frac=1.0 killed learning.")
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
    ap.add_argument("--no-war-nudge", action="store_true",
                    help="Con --auto-support: apaga raid/push/fog-scout. PPO manda la guerra. "
                         "Repair/power/harv/stance/deploy/sell siguen.")
    ap.add_argument("--reset-opt", action="store_true",
                    help="Al --resume, no cargar Adam del ckpt (momentos de una "
                         "política colapsada clavan la cabeza de tipo). Lo pasa "
                         "auto_train tras restaurar best.pt.")
    ap.add_argument("--bc", action="store_true",
                    help="Capa 1: N episodios ScriptedTeacher por iter + NLL BC.")
    ap.add_argument("--bc-only", action="store_true",
                    help="SFT del teacher sin PPO (fase A del onboarding). "
                         "Implica --bc; lambda_bc queda en 1.0.")
    ap.add_argument("--bc-teacher-bot", default=None,
                    choices=("beginner", "easy", "medium", "hard", "brutal",
                             "dummy"),
                    help="Rival del ScriptedTeacher (independiente de --bot-type).")
    ap.add_argument("--bc-games", type=int, default=1,
                    help="Partidas teacher por iter (default 1).")
    ap.add_argument("--bc-rush", type=int, default=0,
                    help="ScriptedTeacher.RUSH_ATTACK_MOVE override (0 = class "
                         "default 8). Fase A lo setea via curriculum a_rush.")
    ap.add_argument("--bc-epochs", type=int, default=1,
                    help="Epochs de NLL BC por iter (default 1).")
    ap.add_argument("--onboard-phase", default=None, choices=("A", "B", "C"),
                    help="Marca la fase A/B/C en metrics.jsonl (lo setea auto_train).")
    ap.add_argument("--eval-games", type=int, default=0,
                    help="En --bc-only: partidas del ALUMNO por iter (wr, sin PPO). "
                         "0 = no mide. Fase A usa 4.")
    ap.add_argument("--sil", action="store_true",
                    help="Capa 1: self-imitation de episodios win/raze>0.")
    ap.add_argument("--bc-warmup", type=int, default=80,
                    help="Iters para bajar lambda_bc de 1.0 a --bc-lambda-end. "
                         "Ignorado en --bc-only.")
    ap.add_argument("--bc-lambda-end", type=float, default=0.0,
                    help="Piso de lambda_bc tras el warmup (default 0). "
                         "Fase B usa 0.25 para no apagar el teacher.")
    ap.add_argument("--bc-keep-incomplete", action="store_true",
                    help="Clonar incomplete largos del teacher (build order). "
                         "Default off (wins-only); solo si se pasa el flag.")
    ap.add_argument("--bc-macro-ticks", type=int, default=0,
                    help="Macro del teacher (0 = --macro-ticks del PPO).")
    ap.add_argument("--bc-max-steps", type=int, default=0,
                    help="max-steps del teacher (0 = --max-steps del PPO).")
    ap.add_argument("--bc-start-iter", type=int, default=0,
                    help="Origen del warmup BC. 0 = ckpt o start_iter. "
                         "No debe resetearse en cada --resume.")
    ap.add_argument("--bc-win-cap", type=int, default=BC_WIN_CAP,
                    help="Cap de steps del TeacherWinBuffer (default 16000, "
                         "~30–40 rushes cortos).")
    ap.add_argument("--bc-win-prefer-ticks", type=int,
                    default=BC_WIN_PREFER_TICKS,
                    help="Wins con ticks>=este se recortan primero y no se "
                         "samplean si hay cortos (default 20000).")
    ap.add_argument("--bc-win-dir", default=None,
                    help="Dir del TeacherWinBuffer (default "
                         "{ckpt_dir}/teacher_wins).")
    ap.add_argument("--bc-replay", action="store_true",
                    help="Fase A/B: no jugar teacher games; BC del ring "
                         "teacher_wins/. auto_train lo pasa cuando las "
                         "cintas ya existen (no en fase C).")
    ap.add_argument("--bc-collect-only", action="store_true",
                    help="Solo recolecta teacher wins al buffer y sale "
                         "(sin SFT/eval/ckpt). auto_train --onboard-collect-only.")
    ap.add_argument("--bc-collect-target", type=int, default=40,
                    help="Con --bc-collect-only: salir 0 al llegar a N "
                         "episodios win en teacher_wins/ (default 40).")
    ap.add_argument("--lambda-sil", type=float, default=0.5,
                    help="Peso SIL cuando --sil (default 0.5).")
    args = ap.parse_args()

    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\nInterrumpido — checkpoint latest.pt conserva la última iter.")


if __name__ == "__main__":
    main()
