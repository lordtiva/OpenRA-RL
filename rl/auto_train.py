#!/usr/bin/env python3
"""
auto_train.py — 1 comando que lanza rl.train y lo vigila (sin 2 ventanas).

- Lanza rl.train --resume latest.pt (o iter0100.pt la primera vez)
- Cada 15s chequea metrics.jsonl; si 5 min (300s) sin avanzar, mata el train
  y lo relanza solo desde latest.pt (no vuelve a 100).
- Si 3 iters seguidas son política muerta (attack-spam, no_op-spam,
  deploy-noop o H<0.15), O wr20 se queda <=0.05 ×5 iters tras un pico
  >=0.20 (sequía, Run 32: H alta, no_op 40%, watchdog viejo no disparaba),
  copia best.pt -> latest.pt, resetea Adam y relanza.
- Si el train termina solo (crash/iters completados), también lo relanza.
- Log en consola + rl/auto_train.log
- train.py sigue escribiendo latest.pt (dashboard + live). Además, tras cada
  iter, si el score es ESTRICTAMENTE mejor, copia latest.pt -> rl/ckpts/best.pt
  (copy, no symlink) + sidecar best.json. Live más adelante: --ckpt rl/ckpts/best.pt

Cuelgues — DOS orígenes distintos (no confundirlos):
  A) Cuelgue del PROCESO PYTHON (train): el .py se traba o muere. Lo cubre
     el watchdog clásico (GPU baja + metrics sin avanzar) -> matar/relanzar.
  B) Cuelgue del DAEMON DOCKER (juego): el contenedor sigue "healthy"
     (/health responde 200) pero el daemon .NET no puede crear sesiones de
     juego -> los logs del contenedor escupen "Session failed to become
     ready" en cascada y el train se queda idle esperando respuestas. Matar
     el train NO arregla nada: el train nuevo se reconecta al mismo daemon
     podrido y vuelve a colgar. La única cura es RECREAR el contenedor
     (docker compose down/up). Este script ahora detecta (B) por los markers
     del log del contenedor y recrea el servicio 'openra-rl' solo.

Uso:  .venv/Scripts/python.exe rl/auto_train.py
      .venv/Scripts/python.exe rl/auto_train.py --scratch
      .venv/Scripts/python.exe rl/auto_train.py --scratch --onboard
      .venv/Scripts/python.exe rl/auto_train.py --onboard --onboard-rewind 24
      .venv/Scripts/python.exe rl/auto_train.py --no-collapse
Ctrl+C para parar todo.

Flags del launcher (no van a rl.train):
  --scratch       pesos random; ignora latest/seed
  --onboard       curriculum A→B→C (SFT teacher vs beginner, PPO+BC beginner,
                  PPO easy). Ver rl/docs/22-onboard.md.
  --onboard-rewind N
                  Una vez, en B: latest <- best/iterN, trunca metrics/race,
                  pinnea BC start. Luego el launch es --onboard.
  --collapse / --no-collapse
                  watchdog COLAPSO (politica muerta) + SEQUIA wr20
                  que restaura best.pt (default: --collapse).
"""
import subprocess, sys, time, pathlib, signal, os, json, re, urllib.request, shutil

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from rl.best_ckpt import (
    DROUGHT_STREAK, dead_policy_reason, is_dead_policy, drought_should_restore,
)
from rl import onboard as ob
CKPT_DIR = ROOT / "rl" / "ckpts"
METRICS = CKPT_DIR / "metrics.jsonl"
CURRICULUM = CKPT_DIR / "curriculum.json"
# Filled in main() when --onboard. launch_train / recover lo leen.
_onboard = None
RESUME_SEED = ROOT / "rl" / "ckpts" / "Run 3 (Full Stack - Asalto)" / "latest.pt"
LOGFILE = ROOT / "rl" / "auto_train.log"
# Capa 1: 4 eps + teacher sequential + PPO+BC+SIL. El primer iter post-launch
# ronda 5-8 min. 300s mataba el update (GPU 30%) antes de escribir metrics.
THRESHOLD_S = 540
# Fase A: 4 teacher + 4 eval alumno. Un iter puede pasar 12-15 min.
ONBOARD_A_THRESHOLD_S = 1200
CHECK_EVERY_S = 15
GPU_LOW_THRESHOLD = 15  # GPU >= esto = collect/update en vuelo, no es cuelgue

# Cuelgue Docker: cada URL de GAME_URLS tiene su servicio compose.
# Recreate toca TODOS los daemons que estaban up, no solo openra-rl.
# Marcas en los logs del contenedor que indican daemon podrido (no Python).
DOCKER_DEAD_MARKERS = (
    "Session failed to become ready",
    "DEADLINE_EXCEEDED",
    "sesión envenenada",
    "session envenenada",
    "gRPC bridge failed to start",
    "OpenRA gRPC bridge failed to start",
)

# 8000 = daemon principal. 8010 = openra-rl-2 (docker-compose.scale.yaml).
# El 5600X rinde 2 daemons x ~2 sesiones, no 8 en uno ni 3 contenedores.
GAME_URLS = (
    "http://localhost:8000",
    "http://localhost:8010",
)
# url, compose files, service name
DAEMONS = (
    ("http://localhost:8000", ("docker-compose.yaml",), "openra-rl"),
    ("http://localhost:8010",
     ("docker-compose.yaml", "docker-compose.scale.yaml"), "openra-rl-2"),
)

# Run 44: seed = Run43 best@80. War nudge OFF. PACK_ARMY = total>=12 (field OK).
# LR 1.0e-4 (era 1.5e-4; clip spikes). 50% easy ancla, 50% PFSP medium+rl.
# Auto-support APM sigue; sin mask naval (lago Singles es legal).
# Macro 50 / max 1000 / gamma 0.995. Overnight: --iters 400 (arranca en 1).
TRAIN_ARGS = [
    sys.executable, "-m", "rl.train",
    "--url", "http://localhost:8000",
    "--iters", "400",
    "--concurrency", "4",
    "--max-steps", "1000",
    "--macro-ticks", "50",
    "--lr", "1.0e-4",
    "--epochs", "1",
    "--clip-eps", "0.15",
    "--max-grad-norm", "1.0",
    "--burn-in", "8",
    "--xf-topk", "16",
    "--qsa-topk", "8",
    "--qsa-block", "8",
    "--batch-size", "128",
    "--scenario", "a_short",
    "--bot-type", "easy",
    "--pfsp",
    "--pfsp-rl",
    "--pfsp-pool", "rl",
    "--pfsp-anchor-prob", "0.5",
    "--shaper-preset", "eradicate_v4",
    "--auto-support",
    "--no-war-nudge",
    # Scratch / Capa 1: BC activo las primeras --bc-warmup iters (lambda 1->0),
    # despues queda SIL. bc-start-iter=1 (no uses 0: train.py trata 0 como
    # unset y en un resume reiniciaria el warmup al start_iter actual).
    #"--bc",
    #"--bc-warmup", "100",
    #"--bc-start-iter", "1",
    "--sil",
    "--lambda-sil", "0.5",
    # Vocab de produccion: ids fijos de roles (scratch / agnostico a faccion).
    "--roles-vocab",
    "--gamma", "0.995",
    "--ckpt-dir", "rl/ckpts",
    "--metrics", "rl/ckpts/metrics.jsonl",
    "--race-file", "rl/ckpts/economy_race.jsonl",
]

def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOGFILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except: pass

def live_game_urls() -> str:
    """Usa cada daemon que responda /health. Si solo esta :8000, un server."""
    ok = []
    for u in GAME_URLS:
        try:
            urllib.request.urlopen(u + "/health", timeout=1.5)
            ok.append(u)
        except Exception:
            pass
    return ",".join(ok) if ok else GAME_URLS[0]


def hang_threshold() -> int:
    # A: teacher SFT. B: 4 PPO + 2 teacher games; primer iter post-launch
    # puede pasar 540s (THRESHOLD_S) antes de escribir metrics.
    if _onboard and _onboard.get("phase") in ("A", "B"):
        return ONBOARD_A_THRESHOLD_S
    return THRESHOLD_S


def find_latest_pt() -> str | None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    latest = CKPT_DIR / "latest.pt"
    if latest.exists():
        return str(latest)
    return None


def find_resume() -> str | None:
    """Resume latest.pt de la raíz de ckpts (lo que mira el dashboard).

    El probe_short quedó archivado en 'Run probe_short-scratch'. Si la raíz
    no tiene latest.pt, cae a Run3. glob de iter*.pt es solo la raíz, no
    las subcarpetas de runs viejos.
    Onboard fase A (primer launch) = tabula rasa, sin caer a Run3.
    """
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if _onboard is not None:
        if not ob.should_resume(_onboard):
            return None
        return find_latest_pt()
    # FORCE_SCRATCH=1 o --scratch en argv: pesos aleatorios, ignora latest/seed.
    if os.environ.get("FORCE_SCRATCH", "").strip() in ("1", "true", "yes") or "--scratch" in sys.argv:
        return None
    hit = find_latest_pt()
    if hit:
        return hit
    pts = sorted(CKPT_DIR.glob("iter*.pt"),
                 key=lambda p: p.stat().st_mtime, reverse=True)
    if pts:
        return str(pts[0])
    if RESUME_SEED.exists():
        return str(RESUME_SEED)
    return None

COLLAPSE_STREAK = 3
COLLAPSE_COOLDOWN_ITERS = 15


def last_metrics_rows(n: int = 3) -> list:
    """Last n unique-by-iter metrics rows (latest write wins per iter)."""
    rows = []
    if not METRICS.exists():
        return rows
    try:
        with open(METRICS, encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                if isinstance(j.get("iter"), int):
                    rows.append(j)
    except OSError:
        return []
    by_iter = {}
    for r in rows:
        by_iter[r["iter"]] = r
    uniq = [by_iter[k] for k in sorted(by_iter)]
    return uniq[-n:]


def restore_best_over_latest() -> bool:
    """Copy best.pt -> latest.pt so the next launch resumes the last good policy.

    Keeps the metrics iteration counter moving forward (patch ckpt['iteration']
    to the last metrics iter) so we don't rewind 220..N and overwrite the new
    run. Fresh Adam is requested via --reset-opt on the next launch, not by
    mutating this file.
    """
    best = CKPT_DIR / "best.pt"
    latest = CKPT_DIR / "latest.pt"
    if not best.exists():
        return False
    tmp = CKPT_DIR / "latest.pt.tmp"
    try:
        shutil.copy2(best, tmp)
        os.replace(tmp, latest)
    except OSError as e:
        log(f"restore best.pt FAIL: {e}")
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        return False
    rows = last_metrics_rows(1)
    keep_iter = int(rows[-1]["iter"]) if rows else None
    if keep_iter is None:
        return True
    try:
        import torch
        ckpt = torch.load(latest, map_location="cpu", weights_only=False)
        ckpt["iteration"] = keep_iter
        torch.save(ckpt, latest)
        log(f"restore: pesos de best.pt, iteration parchada a {keep_iter}")
    except Exception as e:
        log(f"restore: copié best.pt pero no pude parchar iteration ({e})")
    return True


def metrics_mtime() -> float:
    try: return METRICS.stat().st_mtime
    except: return 0

def gpu_util() -> int | None:
    """% GPU util via nvidia-smi, None si no disponible."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, encoding="utf-8", errors="replace", timeout=5)
        # primera GPU si hay varias
        return int(out.strip().splitlines()[0].strip())
    except: return None

# ── Detección/recreación del cuelgue Docker (origen B) ──────────────────────

def docker_available() -> bool:
    try:
        subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=10)
        return True
    except Exception:
        return False

def _compose(files) -> list:
    cmd = ["docker", "compose"]
    for f in files:
        cmd += ["-f", f]
    return cmd


def daemons_up():
    """Daemons cuyo /health responde ahora (los que auto_train estaría usando)."""
    live = set(u.strip() for u in live_game_urls().split(",") if u.strip())
    return [d for d in DAEMONS if d[0] in live] or [DAEMONS[0]]


def _docker_logs_recent(files, service, n: int = 300) -> str:
    """Últimas n líneas de un servicio compose (stdout+stderr)."""
    try:
        return subprocess.check_output(
            _compose(files) + ["logs", "--tail", str(n), service],
            cwd=str(ROOT),
            text=True, encoding="utf-8", errors="replace",
            stderr=subprocess.STDOUT, timeout=20)
    except Exception:
        return ""

def docker_dead_marker_count() -> int:
    """Cuántos markers de daemon podrido hay en logs recientes.

    Un solo 'Session failed' puede ser un retry de 20s; 3+ es cascada.
    """
    if not docker_available():
        return 0
    n = 0
    for url, files, service in daemons_up():
        logs = _docker_logs_recent(files, service, 300)
        if not logs:
            continue
        recent = logs.splitlines()[-150:]
        n += sum(1 for line in recent if any(m in line for m in DOCKER_DEAD_MARKERS))
    return n

def docker_daemon_dead() -> bool:
    """True si hay cascada de markers (3+) en logs recientes."""
    return docker_dead_marker_count() >= 3

def kill_train(proc) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            time.sleep(3)
            if proc.poll() is None:
                proc.kill()
    except Exception:
        pass


def sync_env_into_containers(targets) -> None:
    """Host tree != image. docker cp al container id (no al service name)
    y restart para recargar el Python del env."""
    pairs = [
        (ROOT / "openra_env" / "server" / "openra_environment.py",
         "/app/openra_env/server/openra_environment.py"),
        (ROOT / "openra_env" / "server" / "bridge_client.py",
         "/app/openra_env/server/bridge_client.py"),
        # peer field + FastAdvance peer_commands (image may lag host)
        (ROOT / "openra_env" / "models.py",
         "/app/openra_env/models.py"),
        (ROOT / "openra_env" / "generated" / "rl_bridge_pb2.py",
         "/app/openra_env/generated/rl_bridge_pb2.py"),
        (ROOT / "openra_env" / "generated" / "rl_bridge_pb2_grpc.py",
         "/app/openra_env/generated/rl_bridge_pb2_grpc.py"),
    ]
    for url, files, service in targets:
        try:
            cid = subprocess.check_output(
                _compose(files) + ["ps", "-q", service],
                cwd=str(ROOT), text=True, encoding="utf-8",
                errors="replace", timeout=15).strip()
        except Exception as e:
            log(f"  no container id for {service}: {e}")
            continue
        if not cid:
            log(f"  {service} no está up, skip cp")
            continue
        for src, dst in pairs:
            if not src.exists():
                continue
            r = subprocess.run(
                ["docker", "cp", str(src), f"{cid}:{dst}"],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=30)
            if r.returncode == 0:
                log(f"  docker cp {src.name} -> {service} ({cid[:12]})")
            else:
                err = (r.stderr or r.stdout or "")[:160]
                log(f"  docker cp FAIL {src.name} -> {service}: {err}")
        subprocess.run(["docker", "restart", cid], timeout=60, check=False)
        log(f"  restart {service} (env python recargado)")


def recreate_docker() -> bool:
    """Recrea CADA daemon que estaba up (1 o 2), no solo openra-rl."""
    if not docker_available():
        log("docker no disponible — no puedo recrear el contenedor")
        return False
    targets = daemons_up()
    names = [s for _, _, s in targets]
    log(f"CUELGUE DOCKER detectado — recreando {len(targets)} daemon(s): {names}")
    try:
        for url, files, service in targets:
            log(f"  force-recreate {service} ({url})")
            subprocess.run(
                _compose(files) + ["up", "-d", "--force-recreate", "--no-deps", service],
                cwd=str(ROOT), timeout=240, check=False)
    except Exception as e:
        log(f"error recreando contenedor: {e}")
        return False
    sync_env_into_containers(targets)
    log("esperando /health (max 45s, no 90s fijos)...")
    deadline = time.time() + 45
    while time.time() < deadline:
        ok = []
        for u, _, _ in targets:
            try:
                urllib.request.urlopen(u + "/health", timeout=1.5)
                ok.append(u)
            except Exception:
                pass
        if len(ok) == len(targets):
            log(f"daemons healthy: {','.join(ok)}")
            return True
        time.sleep(5)
    log(f"ADVERTENCIA: /health incompleto tras recreate, live={live_game_urls()}")
    return True

def recover_from_docker_hang(proc) -> subprocess.Popen:
    """Mata el train PRIMERO (evita 1012 mid-recreate), recrea daemons, relanza."""
    log("matando train ANTES de recrear contenedores")
    kill_train(proc)
    recreate_docker()
    time.sleep(2)
    return launch_train()

def _after_onboard_launch():
    """Primer launch de A ya no es scratch; primer launch de C ya hizo --reset-opt."""
    global _onboard
    if not _onboard:
        return
    dirty = False
    if _onboard.get("phase") == "A" and not _onboard.get("a_launched"):
        _onboard["a_launched"] = True
        dirty = True
    if _onboard.get("phase") == "C" and not _onboard.get("c_reset_opt_done"):
        _onboard["c_reset_opt_done"] = True
        dirty = True
    if dirty:
        ob.save_curriculum(CURRICULUM, _onboard)


def last_iter_from_metrics() -> int:
    last_iter = 0
    try:
        with open(METRICS, encoding="utf-8") as _fm:
            for _line in _fm:
                try:
                    _j = json.loads(_line)
                except Exception:
                    continue
                if isinstance(_j.get("iter"), int):
                    last_iter = max(last_iter, _j["iter"])
    except OSError:
        pass
    return last_iter


def try_promote(last_iter: int) -> str | None:
    """Avanza A→B→C→done. Devuelve la fase nueva, o None."""
    global _onboard
    if not _onboard:
        return None
    rows = last_metrics_rows(800)
    nxt = ob.should_promote(_onboard, rows, last_iter, games_per_iter=4)
    if not nxt:
        return None
    old = _onboard.get("phase")
    log(f"onboard PROMOTE {old} -> {nxt} @ iter {last_iter}")
    _onboard["phase"] = nxt
    _onboard["phase_started_iter"] = int(last_iter)
    if nxt == "B":
        ob.append_era_reset(METRICS, "onboard phase B beginner PPO+BC", "beginner")
    elif nxt == "C":
        _onboard["c_reset_opt_done"] = False
        ob.append_era_reset(METRICS, "onboard phase C easy", "easy")
    ob.save_curriculum(CURRICULUM, _onboard)
    return nxt


def launch_train(extra_args=None) -> subprocess.Popen:
    urls = live_game_urls()
    n_srv = urls.count("http")
    extra_args = list(extra_args or [])
    if _onboard is not None:
        args = ob.build_train_argv(list(TRAIN_ARGS), _onboard["phase"], _onboard)
    else:
        args = list(TRAIN_ARGS)
    args[args.index("--url") + 1] = urls
    log(f"servidores de juego: {urls}  ({n_srv} daemon(s); 2do = compose scale openra-rl-2)")
    resume = find_resume()
    iters_s = args[args.index("--iters") + 1] if "--iters" in args else "?"
    preset = (TRAIN_ARGS[TRAIN_ARGS.index("--shaper-preset") + 1]
              if "--shaper-preset" in TRAIN_ARGS else "?")
    phase_tag = (f" onboard={_onboard.get('phase')}" if _onboard else "")
    if resume and pathlib.Path(resume).exists():
        cmd = args + ["--resume", resume] + extra_args
        rel = pathlib.Path(resume)
        try:
            rel = rel.resolve().relative_to(ROOT)
        except Exception:
            pass
        extra_tag = (" " + " ".join(extra_args)) if extra_args else ""
        log(f"LANZANDO train --resume {rel}{extra_tag}{phase_tag}  "
            f"(iters {iters_s}, preset {preset})")
    else:
        cmd = args + extra_args
        log(f"LANZANDO train FROM SCRATCH (pesos aleatorios, sin --resume, "
            f"iters {iters_s}){phase_tag}")
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
    _after_onboard_launch()
    return proc

def parse_auto_args(argv=None):
    """Flags del launcher auto_train (no se reenvian a rl.train)."""
    import argparse
    ap = argparse.ArgumentParser(
        description="Lanza rl.train y lo vigila (cuelgue + colapso opcional).")
    ap.add_argument(
        "--scratch", action="store_true",
        help="Pesos aleatorios: no resume latest/seed.")
    ap.add_argument(
        "--onboard", action="store_true",
        help="Curriculum A→B→C para clonar el repo sin .pt. "
             "Ver rl/docs/22-onboard.md.")
    ap.add_argument("--onboard-sft-iters", type=int, default=20,
                    help="Iters de SFT (fase A).")
    ap.add_argument("--onboard-promote-wr20", type=float, default=0.50,
                    help="wr20 vs beginner para pasar a easy.")
    ap.add_argument("--onboard-done-wr20", type=float, default=0.45,
                    help="wr20 vs easy para marcar DONE.")
    ap.add_argument("--onboard-streak", type=int, default=10,
                    help="Iters consecutivos con wr20 sobre el umbral.")
    ap.add_argument("--onboard-min-iters", type=int, default=20,
                    help="Iters minimos en B/C antes de promover.")
    ap.add_argument(
        "--onboard-rewind", type=int, default=None, metavar="N",
        help="Fase B: copia best/iterN -> latest, trunca metrics y "
             "economy_race a iter<=N, pinnea --bc-start-iter=N. Una vez.")
    ap.add_argument(
        "--collapse", action=argparse.BooleanOptionalAction, default=True,
        help="Watchdog de politica muerta / sequia wr20 que restaura best.pt "
             "(default: on). Usa --no-collapse para desactivarlo.")
    return ap.parse_args(argv)


def _init_onboard(args) -> None:
    """Carga o crea curriculum.json. --scratch --onboard reinicia en A."""
    global _onboard
    if not args.onboard:
        _onboard = None
        return
    overrides = {
        "sft_iters": int(args.onboard_sft_iters),
        "promote_wr20": float(args.onboard_promote_wr20),
        "done_wr20": float(args.onboard_done_wr20),
        "streak": int(args.onboard_streak),
        "min_iters": int(args.onboard_min_iters),
    }
    existing = ob.load_curriculum(CURRICULUM)
    if existing and not args.scratch:
        _onboard = existing
        for k, v in overrides.items():
            _onboard[k] = v
        # Speed knobs: take current defaults so a resume picks up
        # parallel teacher / B mixed-BC without --scratch.
        for k in ("bc_games", "a_macro_ticks", "a_max_steps", "a_k_skip",
                  "a_eval_games",
                  "b_bc_games", "b_bc_epochs", "b_bc_warmup"):
            _onboard[k] = ob.DEFAULTS[k]
        if args.onboard_rewind is not None:
            keep = int(args.onboard_rewind)
            log(f"onboard REWIND a iter {keep}")
            try:
                _onboard, info = ob.rewind_onboard(CKPT_DIR, keep, cfg=_onboard)
            except (FileNotFoundError, ValueError) as e:
                log(f"FAIL: --onboard-rewind {keep}: {e}")
                sys.exit(2)
            log(f"  latest <- {info['src']}  metrics_kept={info['metrics_kept']} "
                f"race_kept={info['race_kept']}  "
                f"b_bc_start_iter={_onboard.get('b_bc_start_iter')}")
        else:
            ob.save_curriculum(CURRICULUM, _onboard)
        log(f"onboard resume phase={_onboard['phase']} "
            f"(sft_iters={_onboard['sft_iters']} "
            f"promote={_onboard['promote_wr20']} "
            f"done={_onboard['done_wr20']})")
        return
    if args.onboard_rewind is not None:
        log("FAIL: --onboard-rewind necesita curriculum.json en fase B "
            "(no uses --scratch).")
        sys.exit(2)
    if existing and args.scratch:
        log("onboard --scratch: reinicia curriculum en fase A")
    elif (CKPT_DIR / "latest.pt").exists() and not args.scratch:
        log("FAIL: --onboard sin curriculum.json y con latest.pt. "
            "Para arrancar de 0: --scratch --onboard "
            "(archivá el run anterior antes).")
        sys.exit(2)
    _onboard = ob.new_curriculum(overrides)
    ob.save_curriculum(CURRICULUM, _onboard)
    ob.append_era_reset(METRICS, "onboard phase A (SFT teacher)", "beginner")
    log(f"onboard START phase A — SFT ScriptedTeacher vs beginner, "
        f"{_onboard['sft_iters']} iters, sin PPO")


def main():
    global _onboard
    args = parse_auto_args()
    _init_onboard(args)
    if args.scratch and not args.onboard:
        os.environ["FORCE_SCRATCH"] = "1"
    collapse_watch = bool(args.collapse)
    _cw = "ON" if collapse_watch else "OFF"
    _sc = "yes" if args.scratch else "no"
    _ob = _onboard.get("phase") if _onboard else "off"
    log(f"auto_train collapse_watch={_cw} scratch={_sc} onboard={_ob}")
    log(f"auto_train iniciado — threshold {hang_threshold()}s, "
        f"check {CHECK_EVERY_S}s (GPU baja en collect NO mata)")
    log(f"métricas={METRICS}  log={LOGFILE}  docker_svc=openra-rl[+openra-rl-2 si up]")
    if docker_available():
        log("docker disponible — se vigilará también el cuelgue del daemon")
    else:
        log("docker NO disponible — solo se vigilará el cuelgue de proceso Python")
    proc: subprocess.Popen | None = None
    last_mtime = metrics_mtime()
    last_progress = time.time()
    gpu_low_streak = 0
    last_restore_iter = 0
    # arranque inicial
    proc = launch_train()
    time.sleep(CHECK_EVERY_S)
    try:
        while True:
            time.sleep(CHECK_EVERY_S)
            # ¿train sigue vivo?
            poll = proc.poll() if proc else None
            if poll is not None:
                last_iter = last_iter_from_metrics()
                if _onboard:
                    nxt = try_promote(last_iter)
                    if nxt == "done":
                        log("ONBOARD DONE — wr20 vs easy en umbral. "
                            "No relanza. Cerrá esta ventana.")
                        sys.exit(0)
                    if nxt:
                        log(f"onboard relanza fase {nxt}")
                        time.sleep(2)
                        proc = launch_train()
                        last_mtime = metrics_mtime()
                        last_progress = time.time()
                        continue
                    if _onboard.get("phase") == "done":
                        log("ONBOARD DONE. Cerrá esta ventana.")
                        sys.exit(0)
                    # Crash o A todavía corta: relanzar la misma fase.
                    # No usar last_iter global vs --iters (metrics sucias de
                    # otro run dirían "completo" en el primer check).
                    log(f"train terminó con exit {poll} — relanzando onboard "
                        f"fase {_onboard.get('phase')} en 5s")
                    time.sleep(5)
                    proc = launch_train()
                    last_mtime = metrics_mtime()
                    last_progress = time.time()
                    continue
                # ¿llegamos a --iters target? No relanzar infinito.
                try:
                    target = int(TRAIN_ARGS[TRAIN_ARGS.index('--iters')+1])
                    if last_iter >= target:
                        log(f"train terminó con exit {poll} — iter {last_iter} >= {target}, COMPLETADO. No relanza.")
                        log("auto_train finalizado correctamente. Cerrá esta ventana.")
                        sys.exit(0)
                except Exception:
                    pass
                log(f"train terminó con exit {poll} — relanzando en 5s")
                time.sleep(5)
                proc = launch_train()
                last_mtime = metrics_mtime()
                last_progress = time.time()
                continue
            # ¿avanzó metrics?
            mtime = metrics_mtime()
            gpu = gpu_util()
            # streak de GPU baja (solo si train vive)
            if gpu is not None:
                if gpu < GPU_LOW_THRESHOLD:
                    gpu_low_streak += 1
                else:
                    gpu_low_streak = 0
            else:
                gpu_low_streak = 0
            gpu_tag = f" GPU {gpu}%" if gpu is not None else ""
            if mtime > last_mtime:
                dt = time.time() - last_progress
                # iter real (max iter) no líneas crudas (hay 4 dup 101/102/103/106 del cuelgue)
                try:
                    iters=[]
                    with open(METRICS, encoding="utf-8") as _f2:
                        for _l in _f2:
                            try: _j=json.loads(_l)
                            except: continue
                            if isinstance(_j.get("iter"), int): iters.append(_j["iter"])
                    n = max(iters) if iters else sum(1 for _ in open(METRICS, encoding="utf-8"))
                except: n = 0
                log(f"ok — metrics avanzó a iter {n} (hace {dt:.0f}s){gpu_tag}")
                last_mtime = mtime
                last_progress = time.time()
                gpu_low_streak = 0
                if _onboard:
                    nxt = try_promote(int(n))
                    if nxt == "done":
                        log("ONBOARD DONE — wr20 vs easy en umbral. "
                            "Matando train y saliendo.")
                        kill_train(proc)
                        sys.exit(0)
                    if nxt:
                        log(f"onboard relanza fase {nxt}")
                        kill_train(proc)
                        time.sleep(2)
                        proc = launch_train()
                        last_mtime = metrics_mtime()
                        last_progress = time.time()
                        gpu_low_streak = 0
                        continue
                collapse_now = collapse_watch
                # A: no wr. B: best@24 iwr=0.5 congela el puntero y
                # deploy-noop cada ~20 iters restaura ese lucky 2/4
                # (loop 52/70/90/129/156). C ya puede usar collapse.
                if _onboard and _onboard.get("phase") in ("A", "B"):
                    collapse_now = False
                if not collapse_now:
                    continue
                rows_tail = last_metrics_rows(max(COLLAPSE_STREAK, DROUGHT_STREAK))
                rows_era = last_metrics_rows(400)
                tail = rows_tail[-COLLAPSE_STREAK:] if len(rows_tail) >= COLLAPSE_STREAK else []
                reasons = [dead_policy_reason(r) for r in tail]
                last_it = int((rows_era or rows_tail or [{"iter": 0}])[-1]["iter"])
                cooldown_ok = (last_it - last_restore_iter) >= COLLAPSE_COOLDOWN_ITERS
                dead = bool(tail) and all(is_dead_policy(r) for r in tail) and cooldown_ok
                drought = cooldown_ok and drought_should_restore(
                    rows_era, last_restore_iter)
                if dead or drought:
                    its = [r["iter"] for r in (tail if dead else rows_era[-DROUGHT_STREAK:])]
                    kind = f"COLAPSO {reasons}" if dead else "SEQUIA wr20"
                    log(f"{kind} — iters {its} — restaurando best.pt -> latest.pt + Adam fresco")
                    kill_train(proc)
                    if restore_best_over_latest():
                        last_restore_iter = last_it
                        log(f"restaurado best.pt sobre latest.pt; cooldown {COLLAPSE_COOLDOWN_ITERS} iters")
                    else:
                        log("no hay best.pt — no pude restaurar, sigo")
                    time.sleep(2)
                    proc = launch_train(extra_args=["--reset-opt"])
                    last_mtime = metrics_mtime()
                    last_progress = time.time()
                    gpu_low_streak = 0
                continue
            idle = time.time() - last_progress
            # GPU baja en collect es NORMAL. Un marker suelto también
            # (retry de 20s). Cascada de 3+ markers = daemon podrido:
            # recrear YA, no esperar 300s con GPU al 1%.
            score = docker_dead_marker_count()
            if score >= 3:
                log(f"daemon podrido — {score} markers, recreando YA (idle {idle:.0f}s){gpu_tag}")
                proc = recover_from_docker_hang(proc)
                last_mtime = metrics_mtime()
                last_progress = time.time()
                gpu_low_streak = 0
                continue
            thr = hang_threshold()
            if idle >= thr:
                if gpu is not None and gpu >= GPU_LOW_THRESHOLD:
                    # Update/inferencia en vuelo: el jsonl se escribe al FINAL
                    # del iter. Matar acá era el loop 636 (teacher 623->135 y
                    # GPU 30% cuando el watchdog cortaba).
                    log(f"en vuelo — idle {idle:.0f}s pero GPU {gpu}%, no mato")
                    last_progress = time.time()
                    continue
                log(f"CUELGUE — idle {idle:.0f}s >= {thr}s{gpu_tag} markers={score}")
                if score >= 1:
                    log("cuelgue es del daemon Docker (no del train) — recreando contenedor")
                    proc = recover_from_docker_hang(proc)
                else:
                    log("cuelgue parece de proceso Python — matando train y relanzando")
                    kill_train(proc)
                    time.sleep(5)
                    proc = launch_train()
                last_mtime = metrics_mtime()
                last_progress = time.time()
                gpu_low_streak = 0
            else:
                extra = f" markers={score}" if score else ""
                log(f"esperando — idle {idle:.0f}s / {hang_threshold()}s{gpu_tag}{extra}")
    except KeyboardInterrupt:
        log("Ctrl+C — terminando train y saliendo")
        if proc and proc.poll() is None:
            try: proc.terminate()
            except: pass
        sys.exit(0)

if __name__ == "__main__":
    main()
