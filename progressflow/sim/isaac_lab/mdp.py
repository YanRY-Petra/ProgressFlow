"""
MDP helpers for ProgressFlow Isaac Lab env.

Keeps observation / success checks close to TableTopSim's observation contract
so Progress Manager + policies stay backend-agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from .scene_cfg import CUBE_NAMES, ZONE_INIT_POS, ZONE_NAMES


def cube_root_pos_w(env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """World-frame root position of a cube, shape (N, 3)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, :3]


def ee_pos_w(env: "ManagerBasedEnv", ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """World-frame end-effector position from FrameTransformer, shape (N, 3)."""
    ee = env.scene[ee_cfg.name]
    # target_frame_pos index 0 == end_effector
    return ee.data.target_pos_w[:, 0, :]


def gripper_open_amount(
    env: "ManagerBasedEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: tuple[str, ...] = ("panda_finger_joint1", "panda_finger_joint2"),
) -> torch.Tensor:
    """Mean finger joint position — larger means more open."""
    robot = env.scene[robot_cfg.name]
    ids, _ = robot.find_joints(list(joint_names))
    return robot.data.joint_pos[:, ids].mean(dim=-1)


def object_in_zone(
    env: "ManagerBasedEnv",
    cube_name: str,
    zone_name: str,
    xy_thresh: float = 0.07,
    z_thresh: float = 0.08,
) -> torch.Tensor:
    """
    Boolean tensor (N,) — True if cube XY is near zone center and cube is near table top.
    """
    color = zone_name.replace("_area", "")
    zone_pos = torch.tensor(ZONE_INIT_POS[color], device=env.device, dtype=torch.float32)
    # Broadcast to num_envs with env origins
    origins = env.scene.env_origins  # (N, 3)
    target = origins + zone_pos.unsqueeze(0)

    cube = env.scene[cube_name]
    pos = cube.data.root_pos_w[:, :3]
    xy_ok = torch.linalg.norm(pos[:, :2] - target[:, :2], dim=-1) < xy_thresh
    z_ok = pos[:, 2] < (target[:, 2] + z_thresh)
    return xy_ok & z_ok


def all_cubes_sorted(env: "ManagerBasedEnv", xy_thresh: float = 0.07) -> torch.Tensor:
    """Success: every color cube lies in its matching color area."""
    ok = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for cube, zone in zip(CUBE_NAMES, ZONE_NAMES):
        ok &= object_in_zone(env, cube, zone, xy_thresh=xy_thresh)
    return ok


def build_tabletop_observation(env: "ManagerBasedEnv", env_id: int = 0) -> dict:
    """
    Build a TableTopSim-compatible observation dict for a single env.

    Used by ProgressFlow policies running on top of Isaac Lab.
    """
    origins = env.scene.env_origins[env_id].detach().cpu()
    cubes = {}
    object_in_zone_map = {}
    for cube_name, zone_name in zip(CUBE_NAMES, ZONE_NAMES):
        pos = env.scene[cube_name].data.root_pos_w[env_id, :3].detach().cpu()
        local = (pos - origins).tolist()
        color = cube_name.replace("_cube", "")
        in_zone = bool(object_in_zone(env, cube_name, zone_name)[env_id].item())
        cubes[cube_name] = {
            "color": color,
            "position": tuple(local),
            "zone": zone_name if in_zone else None,
            "grasped": False,  # refined by controller state
        }
        object_in_zone_map[cube_name] = zone_name if in_zone else None

    zones = {}
    for zone_name in ZONE_NAMES:
        color = zone_name.replace("_area", "")
        zones[zone_name] = {
            "color": color,
            "position": ZONE_INIT_POS[color],
        }

    ee = ee_pos_w(env)[env_id].detach().cpu()
    return {
        "cubes": cubes,
        "zones": zones,
        "grasped_object": None,
        "object_in_zone": object_in_zone_map,
        "last_placed": None,
        "instruction_order": list(CUBE_NAMES),
        "ee_position": tuple((ee - origins).tolist()),
        "gripper_open": float(gripper_open_amount(env)[env_id].item()),
        "backend": "isaac_lab",
    }
