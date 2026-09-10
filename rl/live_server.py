# -*- coding: utf-8 -*-
"""Live broadcaster: HTTP server que expone /api/state para rl/live.html

Uso desde play_vs_checkpoint_live:
    from rl.live_server import LiveBroadcaster
    bc = LiveBroadcaster(port=8765)
    bc.start()
    bc.update({...})

POST /api/recording?episode_id=...  body=webm bytes
  -> <ckpt_dir>/live_recordings/{episode_id}.webm (default rl/ckpts_v2/...)

POST /api/config  JSON {enemy_faction?, spawn?, player_faction?}
  -> queued for the next live episode (opt-in lobby overrides).
GET  /api/config  -> current + pending lobby options.
"""
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

LIVE_HTML = Path(__file__).parent / "live.html"
# repo root = parent of rl/
_REPO = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = _REPO / "rl" / "ckpts_v2" / "live_recordings"

_SAFE_ID = re.compile(r"^[\w.\-]{1,120}$")


class _Handler(BaseHTTPRequestHandler):
    broadcaster = None  # se inyecta

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/state"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            state = self.broadcaster.state if self.broadcaster else {}
            self.wfile.write(json.dumps(state).encode())
            return
        if parsed.path.startswith("/api/config"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            payload = self.broadcaster.config_snapshot() if self.broadcaster else {}
            self.wfile.write(json.dumps(payload).encode())
            return
        if parsed.path in ("/", "/live", "/live.html", "/live.html/"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            html = LIVE_HTML.read_bytes() if LIVE_HTML.exists() else b"<h1>live.html no encontrado</h1>"
            self.wfile.write(html)
            return
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/config":
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length < 0 or length > 16_000:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(b'{"ok":false,"error":"bad body"}')
                return
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw.decode("utf-8") or "{}")
                if not isinstance(data, dict):
                    raise ValueError("config must be an object")
                applied = self.broadcaster.queue_config(data) if self.broadcaster else {}
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode())
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "pending": applied}).encode())
            return
        if parsed.path != "/api/recording":
            self.send_response(404)
            self.end_headers()
            return
        qs = parse_qs(parsed.query)
        eid = (qs.get("episode_id") or ["live"])[0].strip() or "live"
        if not _SAFE_ID.match(eid):
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b'{"ok":false,"error":"bad episode_id"}')
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0 or length > 500_000_000:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b'{"ok":false,"error":"bad body"}')
            return
        body = self.rfile.read(length)
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        out = RECORDINGS_DIR / f"{eid}.webm"
        out.write_bytes(body)
        print(f"[live] recording saved {out} ({len(body)} bytes)", flush=True)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True, "path": str(out)}).encode())

    def log_message(self, format, *args):
        return


class LiveBroadcaster:
    def __init__(self, port: int = 8765):
        self.port = port
        self.state: dict = {"status": "iniciando...", "tick": 0, "done": False, "episode_id": ""}
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # Baseline from CLI; UI may queue overrides for the next episode.
        self.lobby: dict = {
            "player_faction": "RandomAllies",
            "enemy_faction": "Random",
            "spawn": "random",
        }
        self._pending: dict = {}

    def start(self):
        _Handler.broadcaster = self
        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), _Handler)
        self._server.allow_reuse_address = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"[live] visor en http://localhost:{self.port}/  (live.html)")
        print(f"[live] recordings -> {RECORDINGS_DIR}")

    def update(self, patch: dict):
        self.state.update(patch)

    def set_lobby_defaults(self, **kwargs):
        with self._lock:
            for k in ("player_faction", "enemy_faction", "spawn"):
                if k in kwargs and kwargs[k] is not None:
                    self.lobby[k] = kwargs[k]
            self.state["lobby"] = dict(self.lobby)
            if self._pending:
                self.state["lobby_pending"] = dict(self._pending)

    def queue_config(self, data: dict) -> dict:
        """Validate + queue lobby overrides for the next episode."""
        from rl.live_lobby import (
            normalize_enemy_faction,
            normalize_player_faction,
            normalize_spawn,
        )
        pending = {}
        if "enemy_faction" in data and data["enemy_faction"] is not None:
            pending["enemy_faction"] = normalize_enemy_faction(str(data["enemy_faction"]))
        if "player_faction" in data and data["player_faction"] is not None:
            pending["player_faction"] = normalize_player_faction(str(data["player_faction"]))
        if "spawn" in data and data["spawn"] is not None:
            pending["spawn"] = normalize_spawn(str(data["spawn"]))
        if not pending:
            raise ValueError("empty config")
        with self._lock:
            self._pending.update(pending)
            snap = dict(self._pending)
            self.state["lobby_pending"] = snap
        print(f"[live] queued lobby for next episode: {snap}", flush=True)
        return snap

    def take_lobby(self) -> dict:
        """Merge pending UI overrides into lobby defaults and clear pending."""
        with self._lock:
            if self._pending:
                self.lobby.update(self._pending)
                self._pending.clear()
            out = dict(self.lobby)
            self.state["lobby"] = dict(out)
            self.state.pop("lobby_pending", None)
            return out

    def config_snapshot(self) -> dict:
        with self._lock:
            return {
                "lobby": dict(self.lobby),
                "pending": dict(self._pending),
            }

    def stop(self):
        if self._server:
            self._server.shutdown()
