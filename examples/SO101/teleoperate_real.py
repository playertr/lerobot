"""Real robot HAL for SO101 gamepad teleoperation."""

import threading
from teleop_utils import RobotHAL
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


class RealRobotHAL(RobotHAL):
    """Real SO101 robot implementation of RobotHAL."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.robot = None
        self._connected = False
        self._torque_enabled = False
        # Thread-safe pending torque state (set from web thread, applied in main loop)
        self._pending_torque_state = None  # None = no change, True = enable, False = disable
        self._torque_lock = threading.Lock()
    
    def connect(self) -> bool:
        config = SO101FollowerConfig(
            port=self.cfg.robot_port,
            id=self.cfg.robot_id,
            use_degrees=True,
        )
        self.robot = SO101Follower(config)
        
        print(f"\nConnecting to SO101 on {self.cfg.robot_port}...")
        try:
            self.robot.connect()
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
        
        if not self.robot.is_connected:
            print("Robot failed to connect!")
            return False
        
        self._connected = True
        # Start with torque disabled (limp) for safety
        self.disable_torque()
        print("Robot connected! (torque disabled - arm is limp)")
        return True
    
    def disconnect(self):
        if self.robot:
            print("Disconnecting robot...")
            # Ensure torque is disabled before disconnect
            self.disable_torque()
            self.robot.disconnect()
            self._connected = False
    
    def enable_torque(self):
        """Enable torque on all motors - arm becomes powered."""
        if self.robot and self._connected:
            try:
                self.robot.bus.enable_torque()
                self._torque_enabled = True
                print("[Safety] Torque ENABLED - arm is powered")
            except Exception as e:
                print(f"[Safety] Failed to enable torque: {e}")
    
    def disable_torque(self):
        """Disable torque on all motors - arm becomes limp."""
        if self.robot and self._connected:
            try:
                self.robot.bus.disable_torque()
                self._torque_enabled = False
                print("[Safety] Torque DISABLED - arm is limp")
            except Exception as e:
                print(f"[Safety] Failed to disable torque: {e}")
    
    def request_enable_torque(self):
        """Thread-safe request to enable torque (called from web thread)."""
        with self._torque_lock:
            self._pending_torque_state = True
    
    def request_disable_torque(self):
        """Thread-safe request to disable torque (called from web thread)."""
        with self._torque_lock:
            self._pending_torque_state = False
    
    def apply_pending_torque(self):
        """Apply any pending torque state change (called from main loop)."""
        with self._torque_lock:
            pending = self._pending_torque_state
            self._pending_torque_state = None
        
        if pending is True:
            self.enable_torque()
        elif pending is False:
            self.disable_torque()
    
    @property
    def torque_enabled(self) -> bool:
        return self._torque_enabled
    
    def get_observation(self) -> dict:
        if not self._connected:
            return {}
        try:
            return self.robot.get_observation()
        except Exception as e:
            print(f"Read error: {e}")
            return {}
    
    def send_action(self, action: dict):
        if not self._connected or not self._torque_enabled:
            return
        try:
            self.robot.send_action(action)
        except Exception as e:
            print(f"Send error: {e}")
    
    def is_running(self) -> bool:
        return self._connected
    
    def step(self):
        pass
