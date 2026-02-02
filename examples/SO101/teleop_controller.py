"""Teleoperation controller with IK-based motion control."""

from typing import Optional

import numpy as np
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr
from scipy.spatial.transform import Rotation

from lerobot.model.kinematics import RobotKinematics


# SO101 joint names for IK (excludes gripper)
IK_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def pose_to_matrix(position: np.ndarray, rotation: Rotation) -> np.ndarray:
    """Convert position and rotation to 4x4 SE(3) matrix."""
    return pt.transform_from(R=rotation.as_matrix(), p=position)


class TeleoperationController:
    """Handles gamepad input and IK-based teleoperation."""
    
    # Shoulder joint position in world frame (from URDF)
    WORLD_FROM_SHOULDER = pt.transform_from(R=np.eye(3), p=np.array([0.0624, 0.0, 0.0388353]))
    SHOULDER_FROM_WORLD = pt.invert_transform(WORLD_FROM_SHOULDER)
    
    def __init__(self, cfg, kinematics_solver: RobotKinematics):
        self.cfg = cfg
        self.kinematics_solver = kinematics_solver
        
        self.ee_position = np.array([cfg.initial_ee_x, cfg.initial_ee_y, cfg.initial_ee_z])
        self.ee_orientation = Rotation.from_euler('xyz', [0, np.pi/2, 0])
        self.gripper_pos = 50.0
        self._current_joint_obs = None
    
    @staticmethod
    def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Transform a 3D point using SE(3) transform."""
        return pt.transform(transform, pt.vector_to_point(point))[:3]
    
    @staticmethod
    def _transform_direction(transform: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Transform a 3D direction (rotation only, no translation)."""
        return pt.transform(transform, pt.vector_to_direction(direction))[:3]
    
    def initialize_from_observation(self, robot_obs: dict):
        """Initialize EE state from robot's current position via FK."""
        if not robot_obs:
            return
        
        joints = [robot_obs.get(f"{n}.pos") for n in IK_JOINT_NAMES]
        if None in joints:
            return
        
        try:
            ee_pose = self.kinematics_solver.forward_kinematics(np.array(joints))
            self.ee_position = ee_pose[:3, 3].copy()
            self.ee_orientation = Rotation.from_matrix(ee_pose[:3, :3])
            if "gripper.pos" in robot_obs:
                self.gripper_pos = robot_obs["gripper.pos"]
            print(f"Initialized EE: [{self.ee_position[0]:.3f}, {self.ee_position[1]:.3f}, {self.ee_position[2]:.3f}]")
        except Exception as e:
            print(f"Warning: FK init failed: {e}")
    
    def update(self, gamepad, dt: float, current_joint_obs: Optional[dict] = None) -> Optional[dict]:
        """Process gamepad input and return joint commands."""
        self._current_joint_obs = current_joint_obs
        
        self._update_ee_pose(gamepad, dt)
        
        self._clamp_to_bounds()
        self._update_gripper(gamepad, dt)
        
        return self._compute_joint_commands()
    
    def _update_ee_pose(self, gamepad, dt: float):
        """Update EE pose from gamepad input."""
        cfg = self.cfg
        speed, rot_speed = cfg.move_speed, cfg.rot_speed
        world_from_ee = pt.transform_from(R=self.ee_orientation.as_matrix(), p=self.ee_position)
        
        # RB/LB bumpers for forward/back along EE Z-axis (half speed for fine control)
        if gamepad.right_bumper:
            local_z = np.array([0, 0, speed * 0.5 * dt])
            self.ee_position += self._transform_direction(world_from_ee, local_z)
        if gamepad.left_bumper:
            local_z = np.array([0, 0, -speed * 0.5 * dt])
            self.ee_position += self._transform_direction(world_from_ee, local_z)
        
        # Arc sweep around shoulder
        arc_rate = gamepad.left_x * rot_speed * dt
        if abs(arc_rate) > 0.001:
            R_yaw = pr.active_matrix_from_angle(2, -arc_rate)
            ee_in_shoulder = self._transform_point(self.SHOULDER_FROM_WORLD, self.ee_position)
            rotated = R_yaw @ ee_in_shoulder
            new_angle = np.arctan2(rotated[1], rotated[0])
            
            if abs(new_angle) <= np.deg2rad(cfg.ee_max_arc_angle):
                self.ee_position = self._transform_point(self.WORLD_FROM_SHOULDER, rotated)
                self.ee_orientation = Rotation.from_matrix(R_yaw) * self.ee_orientation
        
        # Pitch + Roll
        pitch = gamepad.right_y * rot_speed * dt
        roll = gamepad.right_x * rot_speed * dt
        self.ee_orientation = self.ee_orientation * Rotation.from_euler('zy', [roll, pitch])
        
        # Left stick Y for vertical
        local_up = np.array([gamepad.left_y * speed * dt, 0, 0])
        self.ee_position += self._transform_direction(world_from_ee, local_up)
        
        self._constrain_orientation()
    
    def _clamp_to_bounds(self):
        """Clamp EE position to cylindrical workspace limits."""
        cfg = self.cfg
        
        # Height (world Z)
        self.ee_position[2] = np.clip(self.ee_position[2], cfg.ee_min_height, cfg.ee_max_height)
        
        # Cylindrical bounds in shoulder frame
        ee_in_shoulder = self._transform_point(self.SHOULDER_FROM_WORLD, self.ee_position)
        radius = np.sqrt(ee_in_shoulder[0]**2 + ee_in_shoulder[1]**2)
        angle = np.arctan2(ee_in_shoulder[1], ee_in_shoulder[0])
        
        new_radius = np.clip(radius, cfg.ee_min_radius, cfg.ee_max_radius)
        new_angle = np.clip(angle, -np.deg2rad(cfg.ee_max_arc_angle), np.deg2rad(cfg.ee_max_arc_angle))
        
        if abs(radius - new_radius) > 1e-4 or abs(angle - new_angle) > 1e-4:
            ee_in_shoulder[0] = new_radius * np.cos(new_angle)
            ee_in_shoulder[1] = new_radius * np.sin(new_angle)
            clamped = self._transform_point(self.WORLD_FROM_SHOULDER, ee_in_shoulder)
            self.ee_position[0], self.ee_position[1] = clamped[0], clamped[1]
    
    def _constrain_orientation(self):
        """Constrain EE Z-axis to point radially outward from shoulder."""
        ee_in_shoulder = self._transform_point(self.SHOULDER_FROM_WORLD, self.ee_position)
        radial_dist = np.sqrt(ee_in_shoulder[0]**2 + ee_in_shoulder[1]**2)
        if radial_dist < 0.001:
            return
        
        # Radial direction in world frame
        radial = np.array([ee_in_shoulder[0], ee_in_shoulder[1], 0]) / radial_dist
        radial_world = self._transform_direction(self.WORLD_FROM_SHOULDER, radial)
        
        # Current Z and X axes
        curr_z = self.ee_orientation.apply([0, 0, 1])
        curr_x = self.ee_orientation.apply([1, 0, 0])
        
        # Project Z onto vertical plane containing radial direction
        z_radial = np.dot(curr_z, radial_world)
        z_vertical = curr_z[2]
        new_z = z_radial * radial_world + np.array([0, 0, z_vertical])
        norm = np.linalg.norm(new_z)
        new_z = new_z / norm if norm > 0.001 else radial_world
        
        # Nominal X (perpendicular to Z and world-up)
        world_up = np.array([0, 0, 1])
        nominal_x = np.cross(world_up, new_z)
        x_norm = np.linalg.norm(nominal_x)
        if x_norm < 0.001:
            nominal_x = np.cross(radial_world, new_z)
            x_norm = np.linalg.norm(nominal_x)
        nominal_x = nominal_x / x_norm
        
        # Extract and apply roll
        curr_x_proj = curr_x - np.dot(curr_x, new_z) * new_z
        proj_norm = np.linalg.norm(curr_x_proj)
        if proj_norm > 0.001:
            curr_x_proj /= proj_norm
            cos_roll = np.clip(np.dot(nominal_x, curr_x_proj), -1, 1)
            sin_roll = np.dot(new_z, np.cross(nominal_x, curr_x_proj))
            roll = np.arctan2(sin_roll, cos_roll)
        else:
            roll = 0.0
        
        roll_mat = pr.matrix_from_axis_angle(np.concatenate([new_z, [roll]]))
        new_x = roll_mat @ nominal_x
        new_y = np.cross(new_z, new_x)
        
        self.ee_orientation = Rotation.from_matrix(np.column_stack([new_x, new_y, new_z]))
    
    def _update_gripper(self, gamepad, dt: float):
        if gamepad.left_trigger > 0.1:
            self.gripper_pos = min(100.0, self.gripper_pos + 150.0 * gamepad.left_trigger * dt)
        if gamepad.right_trigger > 0.1:
            self.gripper_pos = max(0.0, self.gripper_pos - 150.0 * gamepad.right_trigger * dt)
    
    def _compute_joint_commands(self) -> Optional[dict]:
        """Run IK and return joint commands."""
        desired_pose = pose_to_matrix(self.ee_position, self.ee_orientation)
        
        if self._current_joint_obs:
            current = np.array([self._current_joint_obs.get(f"{n}.pos", 0.0) for n in IK_JOINT_NAMES])
        else:
            current = np.zeros(len(IK_JOINT_NAMES))
        
        try:
            solution = self.kinematics_solver.inverse_kinematics(
                current_joint_pos=current,
                desired_ee_pose=desired_pose,
                position_weight=1.0,
                orientation_weight=0.1,  # Lower weight to prioritize position accuracy
            )
        except Exception as e:
            print(f"IK failed: {e}")
            self._reset_to_achieved()
            return None
        
        action = {f"{n}.pos": float(solution[i]) for i, n in enumerate(IK_JOINT_NAMES)}
        action["gripper.pos"] = self.gripper_pos
        return action
    
    def _reset_to_achieved(self):
        """Reset target to current FK pose."""
        if not self._current_joint_obs:
            return
        joints = np.array([self._current_joint_obs.get(f"{n}.pos", 0.0) for n in IK_JOINT_NAMES])
        try:
            pose = self.kinematics_solver.forward_kinematics(joints)
            self.ee_position = pose[:3, 3].copy()
            self.ee_orientation = Rotation.from_matrix(pose[:3, :3])
            print("[IK] Reset to achieved pose")
        except Exception as e:
            print(f"[IK] FK reset failed: {e}")
    
    def get_ee_pose(self):
        """Return (position, rotation_matrix)."""
        return self.ee_position.copy(), self.ee_orientation.as_matrix()
