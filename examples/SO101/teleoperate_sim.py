"""MuJoCo simulation HAL for SO101 gamepad teleoperation."""

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from teleop_utils import RobotHAL


def draw_coord_triad(viewer, pos, rot_mat, geom_offset=0, alpha=1.0, axis_length=0.05, axis_radius=0.005):
    """Draw a coordinate frame in MuJoCo viewer."""
    # X axis (red)
    mat_x = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[geom_offset],
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=np.array([axis_radius, axis_length, 0]),
        pos=pos + rot_mat @ np.array([axis_length, 0, 0]),
        mat=(rot_mat @ mat_x).flatten(),
        rgba=np.array([1, 0, 0, alpha])
    )
    # Y axis (green)
    mat_y = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[geom_offset + 1],
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=np.array([axis_radius, axis_length, 0]),
        pos=pos + rot_mat @ np.array([0, axis_length, 0]),
        mat=(rot_mat @ mat_y).flatten(),
        rgba=np.array([0, 1, 0, alpha])
    )
    # Z axis (blue)
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[geom_offset + 2],
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=np.array([axis_radius, axis_length, 0]),
        pos=pos + rot_mat @ np.array([0, 0, axis_length]),
        mat=rot_mat.flatten(),
        rgba=np.array([0, 0, 1, alpha])
    )
    return 3


class SimulationHAL(RobotHAL):
    """MuJoCo simulation implementation of RobotHAL."""
    
    MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    
    def __init__(self, cfg, base_dir: Path):
        self.cfg = cfg
        self.base_dir = base_dir
        self.model = self.data = self.viewer = None
        self._target_ee = self._actual_ee = None
    
    def connect(self) -> bool:
        xml_path = self.base_dir / self.cfg.xml_path if not Path(self.cfg.xml_path).is_absolute() else Path(self.cfg.xml_path)
        if not xml_path.exists():
            print(f"MuJoCo XML not found: {xml_path}")
            return False
        
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        
        # Semi-transparent robot
        for i in range(self.model.ngeom):
            self.model.geom_rgba[i, 3] = 0.5
        
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.azimuth = 45
        self.viewer.cam.elevation = -45
        self.viewer.cam.distance = 1.0
        self.viewer.cam.lookat[:] = [0.15, 0, 0.1]
        
        print("MuJoCo simulation started.")
        return True
    
    def disconnect(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        print("MuJoCo simulation stopped.")
    
    def get_observation(self) -> dict:
        obs = {}
        for name in self.MOTOR_NAMES:
            try:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                val = np.rad2deg(self.data.qpos[self.model.jnt_qposadr[jid]]) if jid != -1 else 0.0
            except Exception:
                val = 0.0
            obs[f"{name}.pos"] = float(val)
        return obs
    
    def send_action(self, action: dict):
        ctrl = np.zeros(self.model.nu)
        for i, name in enumerate(self.MOTOR_NAMES):
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid == -1:
                aid = i
            if f"{name}.pos" in action and aid < len(ctrl):
                ctrl[aid] = np.deg2rad(action[f"{name}.pos"])
        self.data.ctrl[:] = ctrl
    
    def is_running(self) -> bool:
        return self.viewer is not None and self.viewer.is_running()
    
    def step(self):
        # Use control_fps from config
        steps = max(1, int(1 / (self.cfg.control_fps * self.model.opt.timestep)))
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        self._render_ee_frames()
        self.viewer.sync()
    
    def render_ee_frames(self, target_pos, target_rot, actual_pos, actual_rot):
        self._target_ee = (target_pos, target_rot)
        self._actual_ee = (actual_pos, actual_rot)
    
    def _render_ee_frames(self):
        if not self._target_ee:
            return
        
        self.viewer.user_scn.ngeom = 0
        n = draw_coord_triad(self.viewer, np.zeros(3), np.eye(3), 0, 1.0, 0.1, 0.004)
        n += draw_coord_triad(self.viewer, self._target_ee[0], self._target_ee[1], n, 1.0, 0.05, 0.005)
        n += draw_coord_triad(self.viewer, self._actual_ee[0], self._actual_ee[1], n, 0.5, 0.035, 0.003)
        self.viewer.user_scn.ngeom = n
