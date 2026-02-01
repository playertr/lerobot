"""Web server for remote teleoperation control with video streaming."""

import asyncio
import io
import json
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import websockets
from websockets.server import serve
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socket


class WebTeleoperationServer:
    """Serves web UI and handles WebSocket communication for remote control."""
    
    def __init__(self, host: str = "localhost", web_port: int = 8888, 
                 display_host: str = None, urdf_path: str = "so101_new_calib.urdf"):
        # host is what we bind to
        # display_host is for URLs (defaults to host, but use LAN IP when binding to 0.0.0.0)
        self.host = host
        self.display_host = display_host or host
        self.web_port = web_port
        self.urdf_path = urdf_path
        self.static_dir = Path(__file__).parent / "static"
        self.assets_dir = Path(__file__).parent / "assets"
        
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
        self._connected_clients = set()
        self._ws_server = None
        self._http_server = None
        self._http_thread = None
        self._ws_thread = None
        self._loop = None
    
    def get_gamepad_state(self) -> dict:
        """Get current gamepad state (thread-safe)."""
        with self._lock:
            return self._gamepad_state.copy()
    
    def send_camera_frame(self, image: np.ndarray, quality: int = 70):
        """Send camera frame as JPEG to all connected clients."""
        if not self._connected_clients or self._loop is None:
            return
        
        try:
            from PIL import Image
            # Convert to PIL and compress as JPEG
            img = Image.fromarray(image)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=quality)
            jpeg_bytes = buf.getvalue()
            
            # Schedule send on the event loop
            asyncio.run_coroutine_threadsafe(
                self._broadcast_binary(jpeg_bytes),
                self._loop
            )
        except Exception as e:
            pass  # Silently fail if compression fails
    
    def send_joint_positions(self, joint_positions: dict):
        """Send joint positions to all connected clients for 3D visualization."""
        if not self._connected_clients or self._loop is None:
            return
        
        message = json.dumps({
            "type": "joints",
            "positions": joint_positions
        })
        
        asyncio.run_coroutine_threadsafe(
            self._broadcast_text(message),
            self._loop
        )
    
    def send_ee_state(self, target_pos, target_rot, clutch_enabled: bool):
        """Send EE target pose and clutch state to web clients for visualization."""
        if not self._connected_clients or self._loop is None:
            return
        
        # Convert rotation matrix to list for JSON
        message = json.dumps({
            "type": "ee_state",
            "target_pos": target_pos.tolist(),
            "target_rot": target_rot.tolist(),  # 3x3 rotation matrix
            "clutch": clutch_enabled
        })
        
        asyncio.run_coroutine_threadsafe(
            self._broadcast_text(message),
            self._loop
        )
    
    async def _broadcast_binary(self, data: bytes):
        """Send binary data to all connected clients."""
        if self._connected_clients:
            await asyncio.gather(
                *[client.send(data) for client in self._connected_clients],
                return_exceptions=True
            )
    
    async def _broadcast_text(self, message: str):
        """Send text message to all connected clients."""
        if self._connected_clients:
            await asyncio.gather(
                *[client.send(message) for client in self._connected_clients],
                return_exceptions=True
            )
    
    async def _handle_websocket(self, websocket):
        """Handle incoming WebSocket connection."""
        client_addr = websocket.remote_address
        print(f"[Web] Client connected: {client_addr}")
        self._connected_clients.add(websocket)
        
        try:
            # Send a welcome message to confirm connection is working
            await websocket.send(json.dumps({"type": "welcome", "message": "Connected to SO101 server"}))
            print(f"[Web] Sent welcome to {client_addr}")
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "gamepad":
                        with self._lock:
                            self._gamepad_state.update(data.get("state", {}))
                except json.JSONDecodeError:
                    print(f"[Web] Invalid JSON from {client_addr}: {message[:100]}")
            
            print(f"[Web] Message loop ended for {client_addr} (client closed connection)")
        except websockets.exceptions.ConnectionClosedOK:
            print(f"[Web] Connection closed normally: {client_addr}")
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"[Web] Connection closed with error: {client_addr} - code={e.code}, reason={e.reason}")
        except Exception as e:
            print(f"[Web] Unexpected error for {client_addr}: {type(e).__name__}: {e}")
        finally:
            print(f"[Web] Client disconnected: {client_addr}")
            self._connected_clients.discard(websocket)
            # Reset gamepad state on disconnect
            with self._lock:
                for key in self._gamepad_state:
                    if isinstance(self._gamepad_state[key], bool):
                        self._gamepad_state[key] = False
                    else:
                        self._gamepad_state[key] = 0.0
    
    async def _run_websocket_server(self):
        """Run the WebSocket server."""
        self._loop = asyncio.get_event_loop()
        # Increase ping timeout for mobile clients with higher latency
        async with serve(
            self._handle_websocket, 
            self.host, 
            self.web_port + 1,
            ping_interval=30,  # Send ping every 30 seconds
            ping_timeout=60,   # Wait 60 seconds for pong before closing
            close_timeout=10,  # Wait 10 seconds for close handshake
        ) as server:
            self._ws_server = server
            print(f"[Web] WebSocket server running on ws://{self.host}:{self.web_port + 1}")
            await asyncio.Future()  # Run forever
    
    def _run_http_server(self):
        """Run the HTTP server for static files."""
        server_self = self
        
        class Handler(SimpleHTTPRequestHandler):
            def __init__(handler_self, *args, **kwargs):
                super().__init__(*args, directory=str(server_self.static_dir), **kwargs)
            
            def log_message(self, format, *args):
                pass  # Suppress HTTP logs
            
            def do_GET(handler_self):
                # Serve URDF file
                if handler_self.path == f"/{server_self.urdf_path}" or handler_self.path == f"/urdf/{server_self.urdf_path}":
                    urdf_file = Path(__file__).parent / server_self.urdf_path
                    if urdf_file.exists():
                        handler_self.send_response(200)
                        handler_self.send_header("Content-type", "application/xml")
                        handler_self.send_header("Access-Control-Allow-Origin", "*")
                        handler_self.end_headers()
                        handler_self.wfile.write(urdf_file.read_bytes())
                        return
                
                # Serve mesh files from assets directory
                if handler_self.path.startswith("/assets/"):
                    mesh_file = Path(__file__).parent / handler_self.path[1:]  # Remove leading /
                    if mesh_file.exists():
                        handler_self.send_response(200)
                        if mesh_file.suffix.lower() == ".stl":
                            handler_self.send_header("Content-type", "application/octet-stream")
                        else:
                            handler_self.send_header("Content-type", "application/octet-stream")
                        handler_self.send_header("Access-Control-Allow-Origin", "*")
                        handler_self.end_headers()
                        handler_self.wfile.write(mesh_file.read_bytes())
                        return
                
                # Inject config into HTML
                if handler_self.path == "/" or handler_self.path == "/index.html":
                    html_path = server_self.static_dir / "index.html"
                    if html_path.exists():
                        content = html_path.read_text()
                        # Replace config placeholders - use display_host for URLs
                        content = content.replace("{{WS_URL}}", f"ws://{server_self.display_host}:{server_self.web_port + 1}")
                        content = content.replace("{{URDF_PATH}}", f"/{server_self.urdf_path}")
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
        
        print(f"[Web] Remote control UI: http://{self.display_host}:{self.web_port}")
    
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
