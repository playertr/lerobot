"""Rerun visualization for teleoperation."""

import io
from pathlib import Path
from typing import Optional

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from lerobot.model.kinematics import RobotKinematics
from teleop_controller import IK_JOINT_NAMES


# SO101 kinematic chain: joint -> child link
KINEMATIC_CHAIN = [
    ("shoulder_pan", "shoulder_link"),
    ("shoulder_lift", "upper_arm_link"),
    ("elbow_flex", "lower_arm_link"),
    ("wrist_flex", "wrist_link"),
    ("wrist_roll", "gripper_link"),
    ("gripper", "moving_jaw_so101_v1_link"),
]


class RerunVisualizer:
    """Handles Rerun visualization for SO101 robot."""
    
    def __init__(self, urdf_path: str, compress_images: bool = False, minimal: bool = False):
        self.urdf_path = Path(urdf_path)
        self.urdf_prefix = self.urdf_path.stem  # Derive from filename
        self._joint_paths = self._build_joint_paths()
        self._initialized = False
        self._compress_images = compress_images
        self._minimal = minimal  # Only log camera + robot pose
    
    def get_blueprint(self) -> rrb.Blueprint:
        """Create a minimal blueprint showing only camera and 3D robot view."""
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(name="Camera", origin="camera"),
                rrb.Spatial3DView(name="Robot", origin=self.urdf_prefix),
            ),
            collapse_panels=True,
        )
    
    def _build_joint_paths(self) -> dict:
        """Build Rerun entity paths from kinematic chain."""
        paths = {}
        current = f"{self.urdf_prefix}/base_link"
        for joint, link in KINEMATIC_CHAIN:
            current = f"{current}/{joint}/{link}"
            paths[joint] = current
        return paths
    
    def init_visualization(self):
        if self._initialized:
            return
        self._log_urdf()
        self._log_base_frame()
        self._initialized = True
    
    def _log_urdf(self):
        if not self.urdf_path.exists():
            print(f"Warning: URDF not found at {self.urdf_path}")
            return
        try:
            rr.log_file_from_path(self.urdf_path, static=True)
            print(f"Loaded URDF: {self.urdf_path}")
        except Exception as e:
            print(f"Warning: Could not load URDF: {e}")
    
    def _log_base_frame(self):
        rr.log(f"{self.urdf_prefix}/world_frame", rr.Transform3D(translation=[0, 0, 0]), static=True)
        origins = np.zeros((3, 3))
        vectors = np.eye(3) * 0.1
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        rr.log(f"{self.urdf_prefix}/world_frame/axes", 
               rr.Arrows3D(origins=origins, vectors=vectors, colors=colors), static=True)
    
    def log_frame(self, teleop, kinematics_solver: RobotKinematics, robot_obs: dict,
                  joint_action: Optional[dict], gamepad, camera_image: Optional[np.ndarray] = None):
        try:
            # Camera (compressed JPEG for remote, raw for local)
            if camera_image is not None:
                rotated = np.rot90(camera_image, k=-1)
                if self._compress_images:
                    from PIL import Image
                    img = Image.fromarray(rotated)
                    buf = io.BytesIO()
                    img.save(buf, format='JPEG', quality=70)
                    rr.log("camera/image", rr.EncodedImage(contents=buf.getvalue(), media_type="image/jpeg"))
                else:
                    rr.log("camera/image", rr.Image(rotated))
            
            # Joint angles for robot visualization
            if robot_obs:
                self._log_joint_angles(robot_obs)
            
            # Skip the rest in minimal mode
            if self._minimal:
                return
            
            # EE poses
            obs_pos, obs_rot = self.get_observed_ee_pose(kinematics_solver, robot_obs)
            if obs_pos is not None:
                self._log_ee_frame(f"{self.urdf_prefix}/ee_observed", obs_pos, obs_rot,
                                   [255, 165, 0], 0.7, 0.04, 0.008)
            
            target_pos, target_rot = teleop.get_ee_pose()
            self._log_ee_frame(f"{self.urdf_prefix}/ee_target", target_pos, target_rot,
                               [0, 255, 0], 1.0, 0.06, 0.01)
            
            # Status
            rr.log("status/clutch", rr.TextLog(
                f"CLUTCH: {'ENABLED' if teleop.clutch_enabled else 'DISABLED'}"))
            
            # Gamepad
            for attr in ["left_x", "left_y", "right_x", "right_y", "left_trigger", "right_trigger"]:
                rr.log(f"gamepad/{attr}", rr.Scalars(float(getattr(gamepad, attr))))
            
            # Joints
            if joint_action:
                for k, v in joint_action.items():
                    rr.log(f"joints/command/{k.replace('.pos', '')}", rr.Scalars(float(v)))
            
            if robot_obs:
                for k, v in robot_obs.items():
                    if k.endswith(".pos"):
                        rr.log(f"joints/observation/{k.replace('.pos', '')}", rr.Scalars(float(v)))
                self._log_joint_angles(robot_obs)
        except Exception:
            pass
    
    def get_observed_ee_pose(self, kinematics_solver: RobotKinematics, robot_obs: dict):
        if not robot_obs:
            return None, None
        
        joints = [robot_obs.get(f"{n}.pos") for n in IK_JOINT_NAMES]
        if None in joints:
            return None, None
        
        try:
            pose = kinematics_solver.forward_kinematics(np.array(joints))
            return pose[:3, 3], pose[:3, :3]
        except Exception:
            return None, None
    
    def _log_joint_angles(self, joint_positions: dict):
        z_axis = [0, 0, 1]  # All SO101 joints rotate around Z
        for joint, path in self._joint_paths.items():
            val = joint_positions.get(f"{joint}.pos")
            if val is not None:
                rr.log(path, rr.Transform3D(rotation=rr.RotationAxisAngle(axis=z_axis, angle=np.deg2rad(val))))
    
    def _log_ee_frame(self, path: str, pos: np.ndarray, rot: np.ndarray,
                      color: list, alpha: float, axis_len: float, radius: float):
        rr.log(path, rr.Transform3D(translation=pos, mat3x3=rot))
        rr.log(f"{path}/point", rr.Points3D([[0, 0, 0]], colors=[color], radii=[radius]))
        a = int(alpha * 255)
        colors = [[255, 0, 0, a], [0, 255, 0, a], [0, 0, 255, a]]
        rr.log(f"{path}/frame", rr.Arrows3D(
            origins=np.zeros((3, 3)), vectors=np.eye(3) * axis_len, colors=colors))
