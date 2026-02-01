"""Real robot HAL for SO101 gamepad teleoperation."""

from teleop_utils import RobotHAL
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


class RealRobotHAL(RobotHAL):
    """Real SO101 robot implementation of RobotHAL."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.robot = None
        self._connected = False
    
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
        print("Robot connected!")
        return True
    
    def disconnect(self):
        if self.robot:
            print("Disconnecting robot...")
            self.robot.disconnect()
            self._connected = False
    
    def get_observation(self) -> dict:
        if not self._connected:
            return {}
        try:
            return self.robot.get_observation()
        except Exception as e:
            print(f"Read error: {e}")
            return {}
    
    def send_action(self, action: dict):
        if not self._connected:
            return
        try:
            self.robot.send_action(action)
        except Exception as e:
            print(f"Send error: {e}")
    
    def is_running(self) -> bool:
        return self._connected
    
    def step(self):
        pass
