"""Live broadcaster: HTTP server que expone /api/state para rl/live.html

Uso desde play_vs_checkpoint_live:
    from rl.live_server import LiveBroadcaster
    bc = LiveBroadcaster(port=8765)
    bc.start()
    bc.update({...})
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LIVE_HTML = Path(__file__).parent / "live.html"

class _Handler(BaseHTTPRequestHandler):
    broadcaster = None  # se inyecta

    def do_GET(self):
        if self.path.startswith("/api/state"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            state = self.broadcaster.state if self.broadcaster else {}
            self.wfile.write(json.dumps(state).encode())
            return
        # servir live.html en / y /live.html
        if self.path in ("/", "/live", "/live.html", "/live.html/"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = LIVE_HTML.read_bytes() if LIVE_HTML.exists() else b"<h1>live.html no encontrado</h1>"
            self.wfile.write(html)
            return
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"not found")

    def log_message(self, format, *args):
        # silenciar log http
        return


class LiveBroadcaster:
    def __init__(self, port: int = 8765):
        self.port = port
        self.state: dict = {"status": "iniciando...", "tick": 0, "done": False}
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self):
        _Handler.broadcaster = self
        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), _Handler)
        # permitir reuso rápido
        self._server.allow_reuse_address = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"[live] visor en http://localhost:{self.port}/  (live.html)")

    def update(self, patch: dict):
        self.state.update(patch)

    def stop(self):
        if self._server:
            self._server.shutdown()
