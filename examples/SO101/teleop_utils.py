"""Utility classes for teleoperation."""

import threading
import time
from abc import ABC, abstractmethod

import numpy as np


class RobotHAL(ABC):
    """Hardware Abstraction Layer for robot control."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the robot. Returns True on success."""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from the robot."""
        pass
    
    @abstractmethod
    def get_observation(self) -> dict:
        """Get current robot state as {joint_name.pos: degrees}."""
        pass
    
    @abstractmethod
    def send_action(self, action: dict):
        """Send joint commands as {joint_name.pos: degrees}."""
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """Check if the robot/simulation is still running."""
        pass
    
    @abstractmethod
    def step(self):
        """Perform per-frame updates (physics, viewer sync)."""
        pass
    
    def render_ee_frames(self, target_pos: np.ndarray, target_rot: np.ndarray,
                         actual_pos: np.ndarray, actual_rot: np.ndarray):
        """Render EE frames in native viewer. Override in subclasses."""
        pass


class ThreadedCameraWrapper:
    """Non-blocking camera capture via background thread."""
    
    def __init__(self, camera):
        self.camera = camera
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def disconnect(self):
        self.stop()
        if hasattr(self.camera, 'disconnect'):
            self.camera.disconnect()
    
    def _capture_loop(self):
        while self.running:
            try:
                frame = self.camera.async_read()
                with self.lock:
                    self.latest_frame = frame
            except Exception:
                pass
    
    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame
