"""Web server for remote teleoperation control."""

import asyncio
import json
import threading
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote

import websockets
from websockets.server import serve
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socket


class WebTeleoperationServer:
    """Serves web UI and handles WebSocket communication for remote control."""
    
    def __init__(self, host: str = "localhost", web_port: int = 8888, rerun_port: int = 9090, 
                 display_host: str = None):
        # host is what we bind to
        # display_host is for URLs (defaults to host, but use LAN IP when binding to 0.0.0.0)
        self.host = host
        self.display_host = display_host or host
        self.web_port = web_port
        self.rerun_port = rerun_port
        self.static_dir = Path(__file__).parent / "static"
        
        # Gamepad state from remote client
        self._gamepad_state = {
            "left_x": 0.0,
            "left_y": 0.0,
            "right_x": 0.0,
            "right_y": 0.0,
            "left_trigger": 0.0,
            "right_trigger": 0.0,
            "button_a": False,
            "button_b": False,
            "button_x": False,
            "button_y": False,
            "button_lb": False,
            "button_rb": False,
            "dpad_up": False,
            "dpad_down": False,
            "dpad_left": False,
            "dpad_right": False,
        }
        self._lock = threading.Lock()
        self._connected_client = None
        self._ws_server = None
        self._http_server = None
        self._http_thread = None
        self._ws_thread = None
    
    def get_gamepad_state(self) -> dict:
        """Get current gamepad state (thread-safe)."""
        with self._lock:
            return self._gamepad_state.copy()
    
    async def _handle_websocket(self, websocket):
        """Handle incoming WebSocket connection."""
        client_addr = websocket.remote_address
        print(f"[Web] Client connected: {client_addr}")
        self._connected_client = websocket
        
        try:
            # Build URL for Rerun web viewer with gRPC connection
            # The web viewer needs ?url= param to connect to the gRPC server
            # URL-encode the gRPC URL since it contains special characters
            grpc_url = f"rerun+http://{self.display_host}:{self.rerun_port}/proxy"
            web_viewer_url = f"http://{self.display_host}:{self.rerun_port + 1}?url={quote(grpc_url, safe='')}"
            
            await websocket.send(json.dumps({
                "type": "config",
                "rerun_web_url": web_viewer_url
            }))
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "gamepad":
                        with self._lock:
                            self._gamepad_state.update(data.get("state", {}))
                except json.JSONDecodeError:
                    pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            print(f"[Web] Client disconnected: {client_addr}")
            self._connected_client = None
            # Reset gamepad state on disconnect
            with self._lock:
                for key in self._gamepad_state:
                    if isinstance(self._gamepad_state[key], bool):
                        self._gamepad_state[key] = False
                    else:
                        self._gamepad_state[key] = 0.0
    
    async def _run_websocket_server(self):
        """Run the WebSocket server."""
        async with serve(self._handle_websocket, self.host, self.web_port + 1) as server:
            self._ws_server = server
            print(f"[Web] WebSocket server running on ws://{self.host}:{self.web_port + 1}")
            await asyncio.Future()  # Run forever
    
    def _run_http_server(self):
        """Run the HTTP server for static files."""
        class Handler(SimpleHTTPRequestHandler):
            def __init__(handler_self, *args, directory=None, **kwargs):
                super().__init__(*args, directory=str(self.static_dir), **kwargs)
            
            def log_message(self, format, *args):
                pass  # Suppress HTTP logs
            
            def do_GET(handler_self):
                # Inject config into HTML
                if handler_self.path == "/" or handler_self.path == "/index.html":
                    html_path = self.static_dir / "index.html"
                    if html_path.exists():
                        content = html_path.read_text()
                        # Replace config placeholders - use display_host for URLs
                        content = content.replace("{{WS_URL}}", f"ws://{self.display_host}:{self.web_port + 1}")
                        handler_self.send_response(200)
                        handler_self.send_header("Content-type", "text/html")
                        handler_self.end_headers()
                        handler_self.wfile.write(content.encode())
                        return
                super().do_GET()
        
        self._http_server = HTTPServer((self.host, self.web_port), Handler)
        print(f"[Web] HTTP server running on http://{self.host}:{self.web_port}")
        self._http_server.serve_forever()
    
    def start(self):
        """Start both HTTP and WebSocket servers in background threads."""
        # Ensure static directory exists
        self.static_dir.mkdir(exist_ok=True)
        
        # Start HTTP server thread
        self._http_thread = threading.Thread(target=self._run_http_server, daemon=True)
        self._http_thread.start()
        
        # Start WebSocket server thread
        def run_ws():
            asyncio.run(self._run_websocket_server())
        
        self._ws_thread = threading.Thread(target=run_ws, daemon=True)
        self._ws_thread.start()
        
        print(f"[Web] Remote control UI: http://{self.host}:{self.web_port}")
    
    def stop(self):
        """Stop servers."""
        if self._http_server:
            self._http_server.shutdown()
        # WebSocket server stops when thread is killed (daemon)


def get_local_ip() -> str:
    """Get the local IP address for LAN access."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"
