"""
Gamepad teleoperation for SO101 robot arm.

Usage:
    mjpython teleoperate.py                    # Simulation (default)
    mjpython teleoperate.py --sim=False        # Real robot
    mjpython teleoperate.py --remote=True      # Remote control via web UI

Controls:
    X Button:        Toggle CLUTCH (enable/disable motion)
    Left Stick:      Forward/back + Arc sweep around base
    Right Stick:     Pitch + Roll
    D-pad Up/Down:   Move up/down
    LT/RT:           Open/close gripper
    B Button:        Exit

Requirements:
    macOS: pip install hidapi
    Linux: pip install pygame
    Remote: pip install websockets
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import rerun as rr

from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.robot_utils import precise_sleep

from teleop_utils import RobotHAL, ThreadedCameraWrapper
from teleop_controller import TeleoperationController
from teleop_visualizer import RerunVisualizer


@dataclass
class TeleoperateConfig:
    """Configuration for gamepad teleoperation."""
    sim: bool = True
    control_fps: int = 100
    
    # Remote control
    remote: bool = False
    web_host: str = "localhost"  # Use LAN IP for network access
    web_port: int = 8888
    rerun_port: int = 9090
    
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
                     visualizer: RerunVisualizer, camera=None):
    """Main teleoperation control loop."""
    target_dt = 1.0 / cfg.control_fps
    
    robot_obs = robot.get_observation()
    teleop.initialize_from_observation(robot_obs)
    visualizer.init_visualization()
    
    print("\nPress X to enable CLUTCH, then use the gamepad to move the robot.\n")
    
    last_time = time.perf_counter()
    
    try:
        while robot.is_running():
            t0 = time.perf_counter()
            actual_dt = min(t0 - last_time, 0.1)
            last_time = t0
            
            gamepad.update()
            if gamepad.button_b:
                print("\nExit requested.")
                break
            
            robot_obs = robot.get_observation()
            joint_action = teleop.update(gamepad, actual_dt, current_joint_obs=robot_obs)
            
            if joint_action and teleop.clutch_enabled:
                robot.send_action(joint_action)
            
            target_pos, target_rot = teleop.get_ee_pose()
            obs_pos, obs_rot = visualizer.get_observed_ee_pose(kinematics_solver, robot_obs)
            if obs_pos is None:
                obs_pos, obs_rot = target_pos, target_rot
            
            robot.render_ee_frames(target_pos, target_rot, obs_pos, obs_rot)
            
            camera_image = camera.get_latest_frame() if camera else None
            visualizer.log_frame(teleop, kinematics_solver, robot_obs, joint_action, gamepad, camera_image)
            
            robot.step()
            
            elapsed = time.perf_counter() - t0
            precise_sleep(max(target_dt - elapsed, 0.0))
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")


@draccus.wrap()
def main(cfg: TeleoperateConfig):
    """Main entry point."""
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
            rerun_port=cfg.rerun_port,
            display_host=display_host
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
            raw = OpenCVCamera(OpenCVCameraConfig(
                index_or_path=cfg.camera_index, fps=cfg.camera_fps,
                width=cfg.camera_width, height=cfg.camera_height))
            raw.connect()
            camera = ThreadedCameraWrapper(raw)
            camera.start()
            print(f"Camera {cfg.camera_index} connected")
        except Exception as e:
            print(f"Camera failed: {e}")
    
    # Kinematics
    kinematics_solver = RobotKinematics(
        urdf_path=str(urdf_path),
        target_frame_name="gripper_frame_link",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
    )
    
    # Controller & Visualizer
    teleop = TeleoperationController(cfg, kinematics_solver)
    visualizer = RerunVisualizer(str(urdf_path), compress_images=cfg.remote, minimal=cfg.remote)
    
    # Initialize Rerun (serve over WebSocket for remote, spawn viewer locally)
    if cfg.remote:
        rr.init("gamepad_so101_teleop")
        # Start gRPC server with minimal blueprint, connect web viewer to it
        blueprint = visualizer.get_blueprint()
        server_uri = rr.serve_grpc(grpc_port=cfg.rerun_port, default_blueprint=blueprint)
        rr.serve_web_viewer(connect_to=server_uri, web_port=cfg.rerun_port + 1, open_browser=False)
        print(f"[Rerun] gRPC server on {server_uri}")
        print(f"[Rerun] Web viewer at http://{cfg.web_host}:{cfg.rerun_port + 1}")
    else:
        from lerobot.utils.visualization_utils import init_rerun
        init_rerun(session_name="gamepad_so101_teleop")
    
    # Robot HAL
    if cfg.sim:
        from teleoperate_sim import SimulationHAL
        robot = SimulationHAL(cfg, base_dir)
    else:
        from teleoperate_real import RealRobotHAL
        robot = RealRobotHAL(cfg)
    
    if not robot.connect():
        gamepad.disconnect()
        if camera:
            camera.disconnect()
        return
    
    try:
        run_control_loop(cfg, robot, gamepad, teleop, kinematics_solver, visualizer, camera)
    finally:
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
