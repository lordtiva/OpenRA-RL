# -*- coding: utf-8 -*-
"""Live broadcaster: HTTP server que expone /api/state para rl/live.html

Uso desde play_vs_checkpoint_live:
    from rl.live_server import LiveBroadcaster
    bc = LiveBroadcaster(port=8765)
    bc.start()
    bc.update({...})

POST /api/recording?episode_id=...  body=webm bytes
  -> rl/ckpts/live_recordings/{episode_id}.webm
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
RECORDINGS_DIR = _REPO / "rl" / "ckpts" / "live_recordings"

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
        if parsed.path in ("/", "/live", "/live.html", "/live.html/"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = LIVE_HTML.read_bytes() if LIVE_HTML.exists() else b"<h1>live.html no encontrado</h1>"
            self.wfile.write(html)
            return
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"not found")

    def do_POST(self):
        parsed = urlparse(self.path)
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

    def stop(self):
        if self._server:
            self._server.shutdown()
