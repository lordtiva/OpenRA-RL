#!/usr/bin/env python3
"""
auto_train.py — 1 comando que lanza rl.train y lo vigila (sin 2 ventanas).

- Lanza rl.train --resume latest.pt (o iter0100.pt la primera vez)
- Cada 15s chequea metrics.jsonl; si 5 min (300s) sin avanzar, mata el train
  y lo relanza solo desde latest.pt (no vuelve a 100).
- Si el train termina solo (crash/iters completados), también lo relanza.
- Log en consola + rl/auto_train.log

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
import subprocess, sys, time, pathlib, signal, os, json, re

ROOT = pathlib.Path(__file__).resolve().parent.parent
METRICS = ROOT / "rl" / "ckpts" / "metrics.jsonl"
LOGFILE = ROOT / "rl" / "auto_train.log"
THRESHOLD_S = 300  # fallback 5 min si nvidia-smi no está
CHECK_EVERY_S = 15  # más fino para pillar cuelgue rápido
GPU_LOW_THRESHOLD = 15  # % util por debajo = colgado (sano 40-55%)
GPU_LOW_NEEDED = 4  # 4 cheques seguidos = 60s bajo → mata (ahorra ~4 min)

# Cuelgue Docker: servicio y contenedor del juego (docker-compose.yaml).
DOCKER_SERVICE = "openra-rl"
DOCKER_CONTAINER = "openra-rl-openra-rl-1"
# Marcas en los logs del contenedor que indican daemon podrido (no Python).
DOCKER_DEAD_MARKERS = (
    "Session failed to become ready",
    "DEADLINE_EXCEEDED",
    "sesión envenenada",
    "session envenenada",
)

TRAIN_ARGS = [
    sys.executable, "-m", "rl.train",
    "--url", "http://localhost:8000",
    "--iters", "100",
    "--concurrency", "8",
    "--max-steps", "600",
    "--macro-ticks", "80",
    "--lr", "1.5e-4",
    "--batch-size", "128",
    "--scenario", "probe_short",
    "--bot-type", "easy",
    "--shaper-preset", "eradicate_v3",
    "--auto-support",
    "--gamma", "0.995",
]

def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOGFILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except: pass

def find_resume() -> str | None:
    """Regla de resume (intención del usuario):

    - Carpeta rl/ckpts vacía (o inexistente) -> None -> entrenar desde pesos
      aleatorios (from scratch).
    - Carpeta con archivos -> priorizar latest.pt (el que reescribe el train
      cada iter). Si por algún motivo falta latest.pt pero hay otros
      checkpoints (iterNNNN.pt), usar el más nuevo como fallback defensivo
      (no disparar from-scratch y perder progreso). Solo si la carpeta está
      vacía se arranca desde cero.
    """
    ckpt_dir = ROOT / "rl" / "ckpts"
    latest = ckpt_dir / "latest.pt"
    if latest.exists():
        return str(latest)
    if ckpt_dir.is_dir():
        pts = sorted(ckpt_dir.glob("iter*.pt"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
        if pts:
            return str(pts[0])
    return None

def metrics_mtime() -> float:
    try: return METRICS.stat().st_mtime
    except: return 0

def gpu_util() -> int | None:
    """% GPU util via nvidia-smi, None si no disponible."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, timeout=5)
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

def _docker_logs_recent(n: int = 300) -> str:
    """Últimas n líneas del contenedor del juego (stdout+stderr)."""
    try:
        return subprocess.check_output(
            ["docker", "logs", "--tail", str(n), DOCKER_CONTAINER],
            text=True, stderr=subprocess.STDOUT, timeout=15)
    except Exception:
        return ""

def docker_daemon_dead() -> bool:
    """True si el daemon del juego está colgado (cuelgue Docker, no Python).

    El contenedor puede seguir 'healthy' (/health 200) pero sin poder crear
    sesiones. Solo cuenta si el marker aparece en las líneas RECIENTES del
    log (no en historia vieja de horas atrás).
    """
    if not docker_available():
        return False
    logs = _docker_logs_recent(300)
    if not logs:
        return False
    recent = logs.splitlines()[-150:]  # ~últimas 150 líneas
    return any(m in line for line in recent for m in DOCKER_DEAD_MARKERS)

def recreate_docker() -> bool:
    """Recrea el contenedor del juego (compose down/up) para sanear el daemon
    podrido. Devuelve True si el recreate se intentó (el train de host
    reconecta solo al nuevo daemon, pero lo relanzamos igual para limpiar el
    pool de sesiones muertas)."""
    if not docker_available():
        log("docker no disponible — no puedo recrear el contenedor")
        return False
    log("CUELGUE DOCKER detectado — recreando contenedor "
        f"(compose down/up {DOCKER_SERVICE})...")
    try:
        subprocess.run(["docker", "compose", "down", DOCKER_SERVICE],
                       cwd=str(ROOT), timeout=180, check=False)
        subprocess.run(["docker", "compose", "up", "-d", DOCKER_SERVICE],
                       cwd=str(ROOT), timeout=240, check=False)
    except Exception as e:
        log(f"error recreando contenedor: {e}")
        return False
    # Esperar a que el daemon recupere sesiones (start_period + margen).
    log("esperando a que el daemon recupere sesiones (start_period ~90s)...")
    time.sleep(90)
    # Verificar que los logs recientes ya NO muestren markers de cuelgue.
    for _ in range(12):
        if not docker_daemon_dead():
            log("contenedor recreado: daemon sin markers de cuelgue reciente")
            return True
        time.sleep(10)
    log("ADVERTENCIA: el daemon sigue con markers de cuelgue tras recrear")
    return True

def recover_from_docker_hang(proc) -> subprocess.Popen:
    """Si el cuelgue es del daemon Docker, recrea el contenedor y relanza el
    train (limpia el pool de sesiones muertas). Devuelve el nuevo proc."""
    recreate_docker()
    # El contenedor nuevo tiene daemon fresco; matar el train viejo y relanzar
    # para que reconecte limpio (el pool de train reintenta las sesiones).
    try:
        proc.terminate()
        time.sleep(3)
        if proc.poll() is None:
            proc.kill()
    except Exception:
        pass
    time.sleep(5)
    new_proc = launch_train()
    return new_proc

def launch_train() -> subprocess.Popen:
    resume = find_resume()
    if resume and pathlib.Path(resume).exists():
        cmd = TRAIN_ARGS + ["--resume", resume]
        log(f"LANZANDO train --resume {pathlib.Path(resume).name}  (iters {TRAIN_ARGS[TRAIN_ARGS.index('--iters')+1]}, preset {TRAIN_ARGS[TRAIN_ARGS.index('--shaper-preset')+1]})")
    else:
        cmd = TRAIN_ARGS
        log(f"LANZANDO train FROM SCRATCH (pesos aleatorios, sin --resume, iters {TRAIN_ARGS[TRAIN_ARGS.index('--iters')+1]})")
    # cwd=ROOT para que los paths relativos funcionen
    return subprocess.Popen(cmd, cwd=str(ROOT))

def main():
    log(f"auto_train iniciado — threshold {THRESHOLD_S}s, check {CHECK_EVERY_S}s (GPU<{GPU_LOW_THRESHOLD}% x{GPU_LOW_NEEDED} → 60s rápido)")
    log(f"métricas={METRICS}  log={LOGFILE}  docker_svc={DOCKER_SERVICE}")
    if docker_available():
        log("docker disponible — se vigilará también el cuelgue del daemon")
    else:
        log("docker NO disponible — solo se vigilará el cuelgue de proceso Python")
    proc: subprocess.Popen | None = None
    last_mtime = metrics_mtime()
    last_progress = time.time()
    gpu_low_streak = 0
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
                continue
            idle = time.time() - last_progress
            # fast-path GPU: 60s seguidos bajo + idle >60s → cuelgue real
            # (sano nunca está <15% 60s seguidos; colgado está 1-5% fijo)
            if gpu_low_streak >= GPU_LOW_NEEDED and idle >= 60:
                log(f"CUELGUE GPU — idle {idle:.0f}s, GPU {gpu}%<{GPU_LOW_THRESHOLD}% x{gpu_low_streak} ({gpu_low_streak*CHECK_EVERY_S}s)")
                # ¿Es el daemon Docker el que está podrido? Si sí, recrear
                # contenedor (no basta con matar el train).
                if docker_daemon_dead():
                    log("cuelgue es del daemon Docker (no del train) — recreando contenedor")
                    proc = recover_from_docker_hang(proc)
                else:
                    log("cuelgue parece de proceso Python — matando train y relanzando")
                    try:
                        proc.terminate()
                        time.sleep(3)
                        if proc.poll() is None:
                            proc.kill()
                    except: pass
                    time.sleep(5)
                    proc = launch_train()
                last_mtime = metrics_mtime()
                last_progress = time.time()
                gpu_low_streak = 0
                continue
            if idle >= THRESHOLD_S:
                log(f"CUELGUE — idle {idle:.0f}s >= {THRESHOLD_S}s{gpu_tag}")
                # ¿Es el daemon Docker el que está podrido? Si sí, recrear
                # contenedor (no basta con matar el train).
                if docker_daemon_dead():
                    log("cuelgue es del daemon Docker (no del train) — recreando contenedor")
                    proc = recover_from_docker_hang(proc)
                else:
                    log("cuelgue parece de proceso Python — matando train y relanzando")
                    try:
                        proc.terminate()
                        time.sleep(3)
                        if proc.poll() is None:
                            proc.kill()
                    except: pass
                    time.sleep(5)
                    proc = launch_train()
                last_mtime = metrics_mtime()
                last_progress = time.time()
                gpu_low_streak = 0
            else:
                log(f"esperando — idle {idle:.0f}s / {THRESHOLD_S}s{gpu_tag}  (GPU<{GPU_LOW_THRESHOLD}% x{gpu_low_streak}/{GPU_LOW_NEEDED})")
    except KeyboardInterrupt:
        log("Ctrl+C — terminando train y saliendo")
        if proc and proc.poll() is None:
            try: proc.terminate()
            except: pass
        sys.exit(0)

if __name__ == "__main__":
    main()
