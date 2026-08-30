#!/usr/bin/env python3
"""
auto_train.py — 1 comando que lanza rl.train y lo vigila (sin 2 ventanas).

- Lanza rl.train --resume latest.pt (o iter0100.pt la primera vez)
- Cada 15s chequea metrics.jsonl; si 5 min (300s) sin avanzar, mata el train
  y lo relanza solo desde latest.pt (no vuelve a 100).
- Si 3 iters seguidas son política muerta (attack-spam, no_op-spam,
  deploy-noop o H<0.15), copia best.pt -> latest.pt, resetea Adam y relanza.
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
Ctrl+C para parar todo.
"""
import subprocess, sys, time, pathlib, signal, os, json, re, urllib.request, shutil

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from rl.best_ckpt import dead_policy_reason, is_dead_policy
CKPT_DIR = ROOT / "rl" / "ckpts"
METRICS = CKPT_DIR / "metrics.jsonl"
RESUME_SEED = ROOT / "rl" / "ckpts" / "Run 3 (Full Stack - Asalto)" / "latest.pt"
LOGFILE = ROOT / "rl" / "auto_train.log"
# Capa 1: 4 eps + teacher sequential + PPO+BC+SIL. El primer iter post-launch
# ronda 5-8 min. 300s mataba el update (GPU 30%) antes de escribir metrics.
THRESHOLD_S = 540
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

# Dest pasable (SIL -1e9) + rally al dest + stance AttackAnything.
# Resume best.pt 899 vs beginner. No Capa 2. lambda_bc=0.
TRAIN_ARGS = [
    sys.executable, "-m", "rl.train",
    "--url", "http://localhost:8000",
    "--iters", "1200",
    "--concurrency", "4",
    "--max-steps", "624",
    "--macro-ticks", "80",
    "--lr", "1.5e-4",
    "--batch-size", "128",
    "--scenario", "a_short",
    "--bot-type", "beginner",
    "--shaper-preset", "eradicate_v4",
    "--auto-support",
    "--bc",
    "--sil",
    "--bc-warmup", "80",
    "--bc-start-iter", "603",
    "--lambda-sil", "0.5",
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


def find_resume() -> str | None:
    """Resume latest.pt de la raíz de ckpts (lo que mira el dashboard).

    El probe_short quedó archivado en 'Run probe_short-scratch'. Si la raíz
    no tiene latest.pt, cae a Run3. glob de iter*.pt es solo la raíz, no
    las subcarpetas de runs viejos.
    """
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    latest = CKPT_DIR / "latest.pt"
    if latest.exists():
        return str(latest)
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

def launch_train(extra_args=None) -> subprocess.Popen:
    urls = live_game_urls()
    n_srv = urls.count("http")
    args = list(TRAIN_ARGS)
    args[args.index("--url") + 1] = urls
    log(f"servidores de juego: {urls}  ({n_srv} daemon(s); 2do = compose scale openra-rl-2)")
    resume = find_resume()
    extra_args = list(extra_args or [])
    if resume and pathlib.Path(resume).exists():
        cmd = args + ["--resume", resume] + extra_args
        rel = pathlib.Path(resume)
        try:
            rel = rel.resolve().relative_to(ROOT)
        except Exception:
            pass
        extra_tag = (" " + " ".join(extra_args)) if extra_args else ""
        log(f"LANZANDO train --resume {rel}{extra_tag}  (iters {TRAIN_ARGS[TRAIN_ARGS.index('--iters')+1]}, preset {TRAIN_ARGS[TRAIN_ARGS.index('--shaper-preset')+1]})")
    else:
        cmd = args + extra_args
        log(f"LANZANDO train FROM SCRATCH (pesos aleatorios, sin --resume, iters {TRAIN_ARGS[TRAIN_ARGS.index('--iters')+1]})")
    # cwd=ROOT para que los paths relativos funcionen
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env)

def main():
    log(f"auto_train iniciado — threshold {THRESHOLD_S}s, check {CHECK_EVERY_S}s (GPU baja en collect NO mata)")
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
                # ¿llegamos a --iters target? No relanzar infinito.
                try:
                    target = int(TRAIN_ARGS[TRAIN_ARGS.index('--iters')+1])
                    last_iter = 0
                    with open(METRICS, encoding="utf-8") as _fm:
                        for _line in _fm:
                            try: _j = json.loads(_line)
                            except: continue
                            if isinstance(_j.get("iter"), int):
                                last_iter = max(last_iter, _j["iter"])
                    if last_iter >= target:
                        log(f"train terminó con exit {poll} — iter {last_iter} >= {target}, COMPLETADO. No relanza.")
                        log("auto_train finalizado correctamente. Cerrá esta ventana.")
                        sys.exit(0)
                except Exception: pass
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
                rows = last_metrics_rows(COLLAPSE_STREAK)
                reasons = [dead_policy_reason(r) for r in rows]
                if (len(rows) >= COLLAPSE_STREAK
                        and all(is_dead_policy(r) for r in rows)
                        and (int(rows[-1]["iter"]) - last_restore_iter) >= COLLAPSE_COOLDOWN_ITERS):
                    its = [r["iter"] for r in rows]
                    log(f"COLAPSO {reasons} — iters {its} — restaurando best.pt -> latest.pt + Adam fresco")
                    kill_train(proc)
                    if restore_best_over_latest():
                        last_restore_iter = int(rows[-1]["iter"])
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
            if idle >= THRESHOLD_S:
                if gpu is not None and gpu >= GPU_LOW_THRESHOLD:
                    # Update/inferencia en vuelo: el jsonl se escribe al FINAL
                    # del iter. Matar acá era el loop 636 (teacher 623->135 y
                    # GPU 30% cuando el watchdog cortaba).
                    log(f"en vuelo — idle {idle:.0f}s pero GPU {gpu}%, no mato")
                    last_progress = time.time()
                    continue
                log(f"CUELGUE — idle {idle:.0f}s >= {THRESHOLD_S}s{gpu_tag} markers={score}")
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
                log(f"esperando — idle {idle:.0f}s / {THRESHOLD_S}s{gpu_tag}{extra}")
    except KeyboardInterrupt:
        log("Ctrl+C — terminando train y saliendo")
        if proc and proc.poll() is None:
            try: proc.terminate()
            except: pass
        sys.exit(0)

if __name__ == "__main__":
    main()
