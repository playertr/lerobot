"""
Simulated teleoperation using an Xbox/PlayStation gamepad controller.

This example demonstrates gamepad-based teleoperation with MuJoCo:
- Works with Xbox, PlayStation (PS4/PS5), and Logitech controllers
- Uses delta-based control in end-effector frame (body frame)
- Hold clutch to enable motion

Usage:
    python sim_teleoperate_gamepad.py

Controller mapping:
    Left Trigger (LT):   CLUTCH - Hold to enable motion
    Left Stick:          Translate: up/down=forward/back, left/right=left/right (EE frame)
    Right Stick:         Rotate: up/down=pitch down/up, left/right=yaw left/right (EE frame)
    Left Bumper (LB):    Open gripper
    Right Bumper (RB):   Close gripper  
    Right Trigger (RT):  Speed boost (2x)
    B Button:            Exit

Requirements:
    macOS:  pip install hidapi
    Linux:  pip install pygame
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import draccus
import mujoco
import mujoco.viewer
import numpy as np
import rerun as rr

from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from scipy.spatial.transform import Rotation

FPS = 30


class GamepadDirect:
    """
    Direct gamepad reader for teleoperation with custom control scheme.
    
    Control scheme:
    - Left Trigger (LT): Clutch - hold to enable motion
    - Left Stick: Translation in EE frame (forward/back, left/right)
    - Right Stick: Rotation in EE frame (pitch, yaw)
    - Left Bumper (LB): Open gripper
    - Right Bumper (RB): Close gripper
    - Right Trigger (RT): Speed boost
    """
    
    def __init__(self, deadzone: float = 0.15):
        self.deadzone = deadzone
        self.device = None
        
        # Stick values (-1 to 1)
        self.left_x = 0.0
        self.left_y = 0.0
        self.right_x = 0.0
        self.right_y = 0.0
        
        # Triggers (0 to 1)
        self.left_trigger = 0.0
        self.right_trigger = 0.0
        
        # Buttons
        self.left_bumper = False
        self.right_bumper = False
        self.button_b = False
        self.button_start = False
        self._prev_start = False  # For edge detection
        
        # D-pad
        self.dpad_up = False
        self.dpad_down = False
        self.dpad_left = False
        self.dpad_right = False
        
    def connect(self):
        """Connect to gamepad via HID (macOS) or pygame (Linux/Windows)."""
        if sys.platform == "darwin":
            self._connect_hid()
        else:
            self._connect_pygame()
    
    def _connect_hid(self):
        """Connect via HIDAPI for macOS."""
        import hid
        
        devices = hid.enumerate()
        for device in devices:
            name = device.get("product_string", "")
            if any(c in name for c in ["Xbox", "Controller", "Logitech", "PS4", "PS5"]):
                print(f"Found gamepad: {name}")
                self.device = hid.device()
                self.device.open_path(device["path"])
                self.device.set_nonblocking(1)
                self._is_hid = True
                return
        
        raise RuntimeError("No gamepad found. Make sure it's connected.")
    
    def _connect_pygame(self):
        """Connect via pygame for Linux/Windows."""
        import pygame
        pygame.init()
        pygame.joystick.init()
        
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No gamepad found. Make sure it's connected.")
        
        self.device = pygame.joystick.Joystick(0)
        self.device.init()
        self._is_hid = False
        print(f"Found gamepad: {self.device.get_name()}")
    
    def update(self):
        """Read latest gamepad state."""
        if self._is_hid:
            self._update_hid()
        else:
            self._update_pygame()
    
    def _update_hid(self):
        """Update from HID device (macOS Xbox controller)."""
        if not self.device:
            return
        
        # Read multiple times to get stable reading
        data = None
        for _ in range(10):
            d = self.device.read(64)
            if d:
                data = d
        
        if not data or len(data) < 12:
            return
        
        # Xbox controller HID mapping (may vary by controller)
        # Try to parse as Xbox controller
        try:
            # Sticks: 16-bit values, 0-65535, center at 32768
            self.left_x = (int.from_bytes(data[1:3], 'little') - 32768) / 32768.0
            self.left_y = (int.from_bytes(data[3:5], 'little') - 32768) / 32768.0
            self.right_x = (int.from_bytes(data[5:7], 'little') - 32768) / 32768.0
            self.right_y = (int.from_bytes(data[7:9], 'little') - 32768) / 32768.0
            
            # Triggers: 10-bit values, 0-1023
            self.left_trigger = int.from_bytes(data[9:11], 'little') / 1023.0
            self.right_trigger = int.from_bytes(data[11:13], 'little') / 1023.0
            
            # Buttons - Xbox controllers typically have buttons in byte 13-14
            if len(data) > 14:
                buttons1 = data[13]
                buttons2 = data[14]
                
                # Bumpers are in byte 14:
                # Bit 6 (0x40): LB (Left Bumper)
                # Bit 7 (0x80): RB (Right Bumper)
                self.left_bumper = bool(buttons2 & 0x40)
                self.right_bumper = bool(buttons2 & 0x80)
                
                # B button often in byte 13
                self.button_b = bool(buttons1 & 0x02)  # B is usually bit 1
                
                # D-pad is often in byte 13, lower nibble as a hat value
                # Or could be individual bits. Common Xbox: lower 4 bits of byte 13
                dpad = buttons1 & 0x0F
                # Hat values: 0=neutral, 1=up, 2=up-right, 3=right, 4=down-right,
                #             5=down, 6=down-left, 7=left, 8=up-left
                self.dpad_up = dpad in [1, 2, 8]
                self.dpad_down = dpad in [4, 5, 6]
                self.dpad_left = dpad in [6, 7, 8]
                self.dpad_right = dpad in [2, 3, 4]
                
                # X button - bit 2 or 3 in byte 14 (mapped as clutch toggle)
                self.button_start = bool(buttons2 & 0x04) or bool(buttons2 & 0x08)
        except Exception:
            # Fallback: Try Logitech-style 8-bit layout
            self.left_x = (data[1] - 128) / 128.0
            self.left_y = (data[2] - 128) / 128.0
            self.right_x = (data[3] - 128) / 128.0
            self.right_y = (data[4] - 128) / 128.0
            
            # Logitech triggers are often in byte 6
            if len(data) > 6:
                trigger_byte = data[6]
                self.left_trigger = 1.0 if trigger_byte in [4, 6, 12, 14] else 0.0
                self.right_trigger = 1.0 if trigger_byte in [8, 10, 12, 14] else 0.0
                self.left_bumper = trigger_byte in [1, 3, 5, 7, 9, 11, 13, 15]
                self.right_bumper = trigger_byte in [2, 3, 6, 7, 10, 11, 14, 15]
            
            if len(data) > 5:
                buttons = data[5]
                self.button_b = bool(buttons & 0x20)
        
        # Apply deadzone
        self.left_x = 0 if abs(self.left_x) < self.deadzone else self.left_x
        self.left_y = 0 if abs(self.left_y) < self.deadzone else self.left_y
        self.right_x = 0 if abs(self.right_x) < self.deadzone else self.right_x
        self.right_y = 0 if abs(self.right_y) < self.deadzone else self.right_y
    
    def _update_pygame(self):
        """Update from pygame joystick."""
        import pygame
        pygame.event.pump()
        
        # Sticks
        self.left_x = self.device.get_axis(0)
        self.left_y = self.device.get_axis(1)
        self.right_x = self.device.get_axis(2) if self.device.get_numaxes() > 2 else self.device.get_axis(3)
        self.right_y = self.device.get_axis(3) if self.device.get_numaxes() > 3 else self.device.get_axis(4)
        
        # Triggers (axis 4 and 5 on Xbox, or 2 and 5)
        if self.device.get_numaxes() > 5:
            self.left_trigger = (self.device.get_axis(4) + 1) / 2  # Convert -1..1 to 0..1
            self.right_trigger = (self.device.get_axis(5) + 1) / 2
        
        # Bumpers
        self.left_bumper = self.device.get_button(4) if self.device.get_numbuttons() > 4 else False
        self.right_bumper = self.device.get_button(5) if self.device.get_numbuttons() > 5 else False
        self.button_b = self.device.get_button(1) if self.device.get_numbuttons() > 1 else False
        self.button_start = self.device.get_button(7) if self.device.get_numbuttons() > 7 else False
        
        # Apply deadzone
        self.left_x = 0 if abs(self.left_x) < self.deadzone else self.left_x
        self.left_y = 0 if abs(self.left_y) < self.deadzone else self.left_y
        self.right_x = 0 if abs(self.right_x) < self.deadzone else self.right_x
        self.right_y = 0 if abs(self.right_y) < self.deadzone else self.right_y
    
    def check_start_pressed(self) -> bool:
        """Returns True on rising edge of start button press."""
        pressed = self.button_start and not self._prev_start
        self._prev_start = self.button_start
        return pressed
    
    def disconnect(self):
        """Close the gamepad connection."""
        if self.device and self._is_hid:
            self.device.close()


@dataclass
class SimTeleopGamepadConfig:
    """Configuration for gamepad-based simulated teleoperation."""
    xml_path: str = "/Users/timplayer/robots/SO-ARM100/Simulation/SO101/so101_new_calib.xml"
    urdf_path: str = "/Users/timplayer/robots/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
    
    # Movement speed scaling (meters per second at full stick deflection)
    move_speed: float = 0.15
    
    # Rotation speed scaling (radians per second at full stick deflection)
    rot_speed: float = 1.0
    
    # Initial end-effector position (adjust based on your robot's home position)
    initial_ee_x: float = 0.15
    initial_ee_y: float = 0.0
    initial_ee_z: float = 0.15
    
    # End-effector position limits
    ee_min_x: float = 0.05
    ee_max_x: float = 0.35
    ee_min_y: float = -0.25
    ee_max_y: float = 0.25
    ee_min_z: float = 0.02
    ee_max_z: float = 0.35


def pose_to_matrix(position: np.ndarray, rotation: Rotation) -> np.ndarray:
    """Convert position and rotation to a 4x4 transformation matrix."""
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = position
    return matrix


def draw_mj_coord_triad(viewer, pos, rot_mat, geom_offset=0, alpha=1.0, axis_length=0.05, axis_radius=0.005):
    """Draw a coordinate frame in MuJoCo viewer.
    
    Args:
        viewer: MuJoCo viewer instance
        pos: 3D position of the frame origin
        rot_mat: 3x3 rotation matrix
        geom_offset: Starting index for geoms (allows drawing multiple triads)
        alpha: Transparency (1.0 = opaque, 0.0 = transparent)
        axis_length: Length of each axis cylinder
        axis_radius: Radius of each axis cylinder
    """
    # Red cylinder for X axis
    mat_x = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[geom_offset + 0],
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=np.array([axis_radius, axis_length, 0]),
        pos=pos + rot_mat @ np.array([axis_length, 0, 0]),
        mat=(rot_mat @ mat_x).flatten(),
        rgba=np.array([1, 0, 0, alpha])
    )
    
    # Green cylinder for Y axis
    mat_y = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[geom_offset + 1],
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=np.array([axis_radius, axis_length, 0]),
        pos=pos + rot_mat @ np.array([0, axis_length, 0]),
        mat=(rot_mat @ mat_y).flatten(),
        rgba=np.array([0, 1, 0, alpha])
    )
    
    # Blue cylinder for Z axis
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[geom_offset + 2],
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=np.array([axis_radius, axis_length, 0]),
        pos=pos + rot_mat @ np.array([0, 0, axis_length]),
        mat=rot_mat.flatten(),
        rgba=np.array([0, 0, 1, alpha])
    )
    
    return 3  # Number of geoms added


@draccus.wrap()
def main(cfg: SimTeleopGamepadConfig):
    # Load MuJoCo model
    xml_path = Path(cfg.xml_path)
    if not xml_path.exists():
        potential_path = (Path(__file__).parent / cfg.xml_path).resolve()
        if potential_path.exists():
            xml_path = potential_path
        else:
            raise FileNotFoundError(f"MuJoCo XML not found at {xml_path} or {potential_path}")
    
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    
    # Make robot semi-transparent (50% alpha)
    for i in range(model.ngeom):
        model.geom_rgba[i, 3] = 0.5
    
    # Initialize gamepad
    gamepad = GamepadDirect(deadzone=0.15)
    
    print("Connecting to gamepad...")
    print("Make sure your Xbox/PlayStation controller is connected via USB or Bluetooth.")
    print()
    
    try:
        gamepad.connect()
    except Exception as e:
        print(f"Failed to connect gamepad: {e}")
        print("\nTroubleshooting:")
        print("  - Make sure controller is connected and powered on")
        print("  - On macOS: pip install hidapi")
        print("  - On Linux: pip install pygame")
        return
    
    # Initialize kinematics
    urdf_path = Path(cfg.urdf_path)
    if not urdf_path.exists():
        potential_path = (Path(__file__).parent / cfg.urdf_path).resolve()
        if potential_path.exists():
            urdf_path = potential_path
        else:
            raise FileNotFoundError(f"URDF not found at {urdf_path} or {potential_path}")

    motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    # Joint names for IK (excludes gripper - that's controlled separately)
    ik_joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    
    kinematics_solver = RobotKinematics(
        urdf_path=str(urdf_path),
        target_frame_name="gripper_frame_link",
        joint_names=ik_joint_names,
    )

    # Init rerun viewer
    init_rerun(session_name="gamepad_so101_sim_teleop")

    # Initialize end-effector state
    ee_position = np.array([cfg.initial_ee_x, cfg.initial_ee_y, cfg.initial_ee_z])
    ee_orientation = Rotation.from_euler('xyz', [0, np.pi/2, 0])  # Point gripper forward
    gripper_pos = 50.0  # 0 = closed, 100 = open (degrees)
    
    print("\n" + "=" * 60)
    print("Gamepad Teleoperation Started!")
    print("=" * 60)
    print("\nControls:")
    print("  X Button:           Toggle CLUTCH (enable/disable motion)")
    print("  Left Stick U/D:     Move forward/back (EE Z axis)")
    print("  Left Stick L/R:     Sweep around robot base (arc)")
    print("  Right Stick U/D:    Pitch up/down")
    print("  Right Stick L/R:    Roll left/right")
    print("  D-pad Up:           Move up (EE -X)")
    print("  D-pad Down:         Move down (EE +X)")
    print("  Left Trigger (LT):  Open gripper")
    print("  Right Trigger (RT): Close gripper")
    print("  B Button:           Exit")
    print("=" * 60 + "\n")
    
    # Clutch state (toggled with start button)
    clutch_enabled = False
    
    dt = 1.0 / FPS
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set camera to view from behind the robot, looking down
        viewer.cam.azimuth = 45  # 45 degrees to the right of behind
        viewer.cam.elevation = -45  # Higher up, looking down
        viewer.cam.distance = 1.0  # 1 meter away
        viewer.cam.lookat[:] = [0.15, 0, 0.1]  # Look at roughly where the arm is
        
        while viewer.is_running():
            t0 = time.perf_counter()
            
            # Update gamepad
            gamepad.update()
            
            # Check for exit
            if gamepad.button_b:
                print("Exit requested")
                break
            
            # Toggle clutch with start button
            if gamepad.check_start_pressed():
                clutch_enabled = not clutch_enabled
                print(f"Clutch {'ENABLED' if clutch_enabled else 'DISABLED'}")
            
            # Get robot observation from MuJoCo
            robot_obs = {}
            for name in motor_names:
                try:
                    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    if joint_id != -1:
                        qpos_adr = model.jnt_qposadr[joint_id]
                        val_rad = data.qpos[qpos_adr]
                        robot_obs[f"{name}.pos"] = float(np.rad2deg(val_rad))
                    else:
                        robot_obs[f"{name}.pos"] = 0.0
                except Exception:
                    robot_obs[f"{name}.pos"] = 0.0

            # Only update position/orientation when clutch is enabled
            if clutch_enabled:
                # Get speed
                speed = cfg.move_speed
                rot_speed = cfg.rot_speed
                
                ee_rot_mat = ee_orientation.as_matrix()
                
                # === Left stick: Forward/back translation + Arc sweep ===
                # EE frame convention: X=down, Y=left, Z=front
                # Up/down on stick → translate along EE +Z/-Z (forward/back)
                local_z = -gamepad.left_y * speed * dt  # Stick up → +Z (forward)
                world_delta = ee_rot_mat @ np.array([0, 0, local_z])
                ee_position += world_delta
                
                # Left/right on stick → sweep EE around robot base (world Z axis)
                # This rotates both position and orientation about world vertical
                arc_rate = gamepad.left_x * rot_speed * dt  # Stick right → rotate clockwise (from above)
                if abs(arc_rate) > 0.001:
                    # Rotate position around world Z axis (at origin)
                    world_yaw_rot = Rotation.from_euler('z', -arc_rate)
                    new_position = world_yaw_rot.apply(ee_position)
                    
                    # Only apply if position won't be clamped (couple position and orientation)
                    would_clamp = (
                        new_position[0] < cfg.ee_min_x or new_position[0] > cfg.ee_max_x or
                        new_position[1] < cfg.ee_min_y or new_position[1] > cfg.ee_max_y or
                        new_position[2] < cfg.ee_min_z or new_position[2] > cfg.ee_max_z
                    )
                    if not would_clamp:
                        ee_position = new_position
                        # Also rotate the EE orientation to maintain relative heading
                        ee_orientation = world_yaw_rot * ee_orientation
                
                # === Right stick: Pitch + Roll ===
                # Up/down → pitch (about EE Y axis)
                pitch_delta = -gamepad.right_y * rot_speed * dt  # Stick up → pitch down (+Y)
                # Left/right → roll (about EE Z axis)
                roll_delta = gamepad.right_x * rot_speed * dt    # Stick right → roll right (+Z)
                
                # Apply rotation deltas in EE frame (local rotation)
                delta_rot = Rotation.from_euler('zy', [roll_delta, pitch_delta])
                ee_orientation = ee_orientation * delta_rot
                
                # === D-pad: Vertical motion (translate along EE X axis) ===
                if gamepad.dpad_up:  # Move up (-X in EE frame)
                    vert_delta = ee_rot_mat @ np.array([-speed * dt, 0, 0])
                    ee_position += vert_delta
                if gamepad.dpad_down:  # Move down (+X in EE frame)
                    vert_delta = ee_rot_mat @ np.array([speed * dt, 0, 0])
                    ee_position += vert_delta
            
            # Clamp position to bounds
            ee_position[0] = np.clip(ee_position[0], cfg.ee_min_x, cfg.ee_max_x)
            ee_position[1] = np.clip(ee_position[1], cfg.ee_min_y, cfg.ee_max_y)
            ee_position[2] = np.clip(ee_position[2], cfg.ee_min_z, cfg.ee_max_z)
            
            # Handle gripper with triggers (always active, not just with clutch)
            if gamepad.left_trigger > 0.1:  # Open
                gripper_pos = min(100.0, gripper_pos + 150.0 * gamepad.left_trigger * dt)
            if gamepad.right_trigger > 0.1:  # Close
                gripper_pos = max(0.0, gripper_pos - 150.0 * gamepad.right_trigger * dt)
            
            # Build desired EE pose as 4x4 matrix
            desired_pose = pose_to_matrix(ee_position, ee_orientation)
            
            # Get current joint positions for IK initial guess (exclude gripper)
            current_joints_deg = np.array([
                robot_obs.get(f"{name}.pos", 0.0) for name in ik_joint_names
            ])
            
            # Run IK
            try:
                ik_solution_deg = kinematics_solver.inverse_kinematics(
                    current_joint_pos=current_joints_deg,
                    desired_ee_pose=desired_pose,
                    position_weight=1.0,
                    orientation_weight=0.1,
                )
            except Exception as e:
                print(f"IK failed: {e}")
                viewer.sync()
                precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))
                continue
            
            # Build joint action dict
            joint_action = {}
            for i, name in enumerate(ik_joint_names):
                joint_action[f"{name}.pos"] = float(ik_solution_deg[i])
            joint_action["gripper.pos"] = gripper_pos

            # Get actual EE pose from forward kinematics
            actual_ee_pose = kinematics_solver.forward_kinematics(current_joints_deg)
            actual_ee_pos = actual_ee_pose[:3, 3]
            actual_ee_rot = actual_ee_pose[:3, :3]

            # Visualize both EE poses in MuJoCo viewer
            viewer.user_scn.ngeom = 0
            
            # Draw robot base frame (world origin, identity rotation)
            base_rot_mat = np.eye(3)
            ngeom = draw_mj_coord_triad(
                viewer, np.array([0.0, 0.0, 0.0]), base_rot_mat,
                geom_offset=0, alpha=0.8, axis_length=0.08, axis_radius=0.006
            )
            
            # Draw target EE pose (bright, larger)
            target_rot_mat = ee_orientation.as_matrix()
            ngeom += draw_mj_coord_triad(
                viewer, ee_position, target_rot_mat, 
                geom_offset=ngeom, alpha=1.0, axis_length=0.05, axis_radius=0.005
            )
            
            # Draw actual EE pose (dimmer, smaller)
            ngeom += draw_mj_coord_triad(
                viewer, actual_ee_pos, actual_ee_rot,
                geom_offset=ngeom, alpha=0.5, axis_length=0.035, axis_radius=0.003
            )
            
            viewer.user_scn.ngeom = ngeom

            # Log to rerun
            rr.log("world/origin", rr.Transform3D(translation=[0, 0, 0]))
            rr.log(
                "world/target_pose",
                rr.Transform3D(
                    translation=ee_position,
                    mat3x3=target_rot_mat
                )
            )
            rr.log(
                "world/actual_pose",
                rr.Transform3D(
                    translation=actual_ee_pos,
                    mat3x3=actual_ee_rot
                )
            )

            # Apply action to MuJoCo
            ctrl = np.zeros(model.nu)
            for i, name in enumerate(motor_names):
                actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                if actuator_id == -1:
                    actuator_id = i
                
                if f"{name}.pos" in joint_action:
                    val_deg = joint_action[f"{name}.pos"]
                    if actuator_id < len(ctrl):
                        ctrl[actuator_id] = np.deg2rad(val_deg)
            
            data.ctrl[:] = ctrl
            
            # Step simulation
            steps = max(1, int(1 / (FPS * model.opt.timestep)))
            for _ in range(steps):
                mujoco.mj_step(model, data)
            
            # Visualize
            viewer.sync()
            
            # Rerun logging
            gamepad_state = {
                "clutch": clutch_enabled,
                "left_x": gamepad.left_x,
                "left_y": gamepad.left_y,
                "right_x": gamepad.right_x,
                "right_y": gamepad.right_y,
                "left_trigger": gamepad.left_trigger,
                "right_trigger": gamepad.right_trigger,
            }
            log_rerun_data(
                observation={"gamepad": gamepad_state},
                action=dict(joint_action)
            )

            precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))
    
    gamepad.disconnect()
    print("\nTeleoperation ended.")


if __name__ == "__main__":
    main()
