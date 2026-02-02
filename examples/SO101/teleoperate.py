"""
Gamepad teleoperation for SO101 robot arm.

Usage:
    mjpython teleoperate.py                    # Simulation (default)
    mjpython teleoperate.py --sim=False        # Real robot
    mjpython teleoperate.py --remote=True      # Remote control via web UI (mobile-friendly)

Controls:
    Left Stick X:    Arc sweep around base
    Left Stick Y:    Move up/down
    Right Stick:     Pitch + Roll
    LB/RB:           Peck forward/back along EE Z-axis
    LT/RT:           Open/close gripper

Requirements:
    macOS: pip install hidapi
    Linux: pip install pygame
    Remote: pip install websockets pillow
"""

import time
import signal
import atexit
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.robot_utils import precise_sleep

from teleop_utils import RobotHAL, ThreadedCameraWrapper
from teleop_controller import TeleoperationController
from teleop_visualizer import RerunVisualizer


# Global reference for signal handlers
_cleanup_robot = None


@dataclass
class TeleoperateConfig:
    """Configuration for gamepad teleoperation."""
    sim: bool = True
    headless: bool = False  # For servers without display (remote mode auto-enables)
    control_fps: int = 100
    
    # Remote control
    remote: bool = False
    web_host: str = "localhost"  # Use 0.0.0.0 for LAN access
    web_port: int = 8888
    
    # Model paths (relative to this file)
    urdf_path: str = "so101_new_calib.urdf"
    xml_path: str = "so101_new_calib.xml"
    
    # Real robot
    robot_port: str = "/dev/tty.usbmodem5AB01831681"
    robot_id: str = "so101_follower"
    
    # Camera (set camera_index to enable)
    camera_index: Optional[int] = None
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    
    # Motion
    move_speed: float = 0.15  # m/s at full stick
    rot_speed: float = 1.0    # rad/s at full stick
    
    # Initial EE position
    initial_ee_x: float = 0.15
    initial_ee_y: float = 0.0
    initial_ee_z: float = 0.15
    
    # Workspace limits (cylindrical, centered on shoulder)
    ee_min_height: float = -0.02
    ee_max_height: float = 0.35
    ee_min_radius: float = 0.05
    ee_max_radius: float = 0.4
    ee_max_arc_angle: float = 85.0  # degrees from center


def run_control_loop(cfg: TeleoperateConfig, robot: RobotHAL, gamepad,
                     teleop: TeleoperationController, kinematics_solver: RobotKinematics,
                     visualizer, camera=None, web_server=None):
    """Main teleoperation control loop."""
    target_dt = 1.0 / cfg.control_fps
    stream_interval = 1.0 / 15  # Stream at 15 FPS for web
    last_stream_time = 0
    
    robot_obs = robot.get_observation()
    teleop.initialize_from_observation(robot_obs)
    if visualizer:
        visualizer.init_visualization()
    
    if cfg.remote:
        print("\n[Safety] Robot starts DISABLED (limp). Use web UI to:")
        print("  1. Click 'Enable Robot' to power the arm")
        print("  2. Use the gamepad to move the robot\n")
    else:
        print("\nUse the gamepad to move the robot.\n")
    
    last_time = time.perf_counter()
    was_torque_enabled = getattr(robot, 'torque_enabled', True)  # Track previous state
    
    try:
        while robot.is_running():
            t0 = time.perf_counter()
            actual_dt = min(t0 - last_time, 0.1)
            last_time = t0
            
            # Apply any pending torque state changes from web thread
            if hasattr(robot, 'apply_pending_torque'):
                robot.apply_pending_torque()
            
            # Check for torque state transition: disabled -> enabled
            # Reset EE target to current observed pose to prevent sudden jumps
            is_torque_enabled = getattr(robot, 'torque_enabled', True)
            if is_torque_enabled and not was_torque_enabled:
                robot_obs = robot.get_observation()
                teleop.initialize_from_observation(robot_obs)
                print("[Safety] EE target reset to current position on enable")
            was_torque_enabled = is_torque_enabled
            
            gamepad.update()
            
            robot_obs = robot.get_observation()
            joint_action = teleop.update(gamepad, actual_dt, current_joint_obs=robot_obs)
            
            if joint_action:
                robot.send_action(joint_action)
            
            target_pos, target_rot = teleop.get_ee_pose()
            
            robot.render_ee_frames(target_pos, target_rot, target_pos, target_rot)
            
            camera_image = camera.get_latest_frame() if camera else None

            # Stream to web clients (at reduced rate)
            if web_server and t0 - last_stream_time >= stream_interval:
                last_stream_time = t0
                # Stream camera image
                if camera_image is not None:
                    rotated = np.rot90(camera_image, k=-1)
                    web_server.send_camera_frame(rotated, quality=60)
                # Stream joint positions for 3D visualization
                if robot_obs:
                    joint_positions = {k.replace('.pos', ''): v for k, v in robot_obs.items() if k.endswith('.pos')}
                    web_server.send_joint_positions(joint_positions)
                # Stream EE target pose for visualization
                web_server.send_ee_state(target_pos, target_rot)
            
            # Log to Rerun (local visualization only in non-remote mode)
            if visualizer:
                visualizer.log_frame(teleop, kinematics_solver, robot_obs, joint_action, gamepad, camera_image)
            
            robot.step()
            
            elapsed = time.perf_counter() - t0
            precise_sleep(max(target_dt - elapsed, 0.0))
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")


@draccus.wrap()
def main(cfg: TeleoperateConfig):
    """Main entry point."""
    global _cleanup_robot
    
    base_dir = Path(__file__).parent
    urdf_path = base_dir / cfg.urdf_path if not Path(cfg.urdf_path).is_absolute() else Path(cfg.urdf_path)
    
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    
    # Web server for remote control
    web_server = None
    if cfg.remote:
        from teleop_web import WebTeleoperationServer, get_local_ip
        
        host = cfg.web_host
        # When binding to 0.0.0.0, detect LAN IP for URLs
        if host == "0.0.0.0":
            display_host = get_local_ip()
            print(f"[Web] Binding to all interfaces, LAN IP: {display_host}")
        else:
            display_host = host
        
        web_server = WebTeleoperationServer(
            host=host,
            web_port=cfg.web_port,
            display_host=display_host,
            urdf_path=cfg.urdf_path
        )
        web_server.start()
        
        if host == "0.0.0.0":
            print(f"[Web] Access from other devices: http://{display_host}:{cfg.web_port}")
    
    # Gamepad (local or remote)
    print("Connecting to gamepad...")
    if cfg.remote:
        from teleop_remote_gamepad import RemoteGamepadController
        gamepad = RemoteGamepadController(web_server, deadzone=0.15)
    else:
        from teleop_gamepad import GamepadController
        gamepad = GamepadController(deadzone=0.15)
    
    try:
        gamepad.connect()
    except Exception as e:
        print(f"Gamepad failed: {e}")
        if not cfg.remote:
            return
    
    # Camera
    camera = None
    if cfg.camera_index is not None:
        try:
            from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
            # Use integer index for macOS, device path for Linux
            import platform
            if platform.system() == "Darwin":
                camera_path = cfg.camera_index  # macOS uses integer index
            else:
                camera_path = f"/dev/video{cfg.camera_index}" if isinstance(cfg.camera_index, int) else cfg.camera_index
            raw = OpenCVCamera(OpenCVCameraConfig(
                index_or_path=camera_path, fps=cfg.camera_fps,
                width=cfg.camera_width, height=cfg.camera_height))
            raw.connect()
            camera = ThreadedCameraWrapper(raw)
            camera.start()
            print(f"Camera {camera_path} connected")
        except Exception as e:
            print(f"Camera failed: {e}")
    
    # Kinematics
    kinematics_solver = RobotKinematics(
        urdf_path=str(urdf_path),
        target_frame_name="gripper_frame_link",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
    )
    
    # Controller
    teleop = TeleoperationController(cfg, kinematics_solver)
    
    # Visualizer (Rerun for local mode only)
    visualizer = None
    if not cfg.remote:
        import rerun as rr
        from lerobot.utils.visualization_utils import init_rerun
        init_rerun(session_name="gamepad_so101_teleop")
        visualizer = RerunVisualizer(str(urdf_path))
    
    # Robot HAL
    if cfg.sim:
        from teleoperate_sim import SimulationHAL
        # Auto-enable headless for remote mode on Linux (no DISPLAY)
        import os
        headless = cfg.headless or (cfg.remote and not os.environ.get('DISPLAY'))
        robot = SimulationHAL(cfg, base_dir, headless=headless)
    else:
        from teleoperate_real import RealRobotHAL
        robot = RealRobotHAL(cfg)
    
    if not robot.connect():
        gamepad.disconnect()
        if camera:
            camera.disconnect()
        return
    
    # For non-remote mode, auto-enable torque after connection
    # In remote mode, user must enable via web UI for safety
    if not cfg.remote and hasattr(robot, 'enable_torque'):
        robot.enable_torque()
    
    # Safety: Set up signal handlers and atexit for graceful shutdown
    # This ensures torque is disabled even on SIGTERM/SIGINT
    def cleanup_on_exit():
        """Ensure robot is disabled on exit."""
        print("\n[Safety] Cleanup: disabling robot torque...")
        if hasattr(robot, 'disable_torque'):
            robot.disable_torque()
    
    _cleanup_robot = cleanup_on_exit
    atexit.register(cleanup_on_exit)
    
    def signal_handler(signum, frame):
        """Handle SIGTERM/SIGINT gracefully."""
        print(f"\n[Safety] Received signal {signum}, shutting down...")
        cleanup_on_exit()
        raise SystemExit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Wire up web server safety callbacks (for real robot only)
    # Use request_* methods for thread safety - actual changes applied in main loop
    if web_server and hasattr(robot, 'request_enable_torque'):
        web_server.set_enable_callback(robot.request_enable_torque)
        web_server.set_disable_callback(robot.request_disable_torque)
    
    try:
        run_control_loop(cfg, robot, gamepad, teleop, kinematics_solver, visualizer, camera, web_server)
    finally:
        # Ensure cleanup runs
        if hasattr(robot, 'disable_torque'):
            robot.disable_torque()
        robot.disconnect()
        if hasattr(gamepad, 'disconnect'):
            gamepad.disconnect()
        if camera:
            camera.disconnect()
        if web_server:
            web_server.stop()
        print("\nTeleoperation ended.")


if __name__ == "__main__":
    main()
