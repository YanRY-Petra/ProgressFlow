"""
Isaac Lab scene configuration for ProgressFlow.

Scene layout
------------
Isaac Lab → Franka Panda → Table → 3 Cubes → 3 Target Zones

Cube spawn (near side of table):
  red_cube   @ (0.45, -0.20, 0.0203)
  blue_cube  @ (0.55, -0.20, 0.0203)
  green_cube @ (0.65, -0.20, 0.0203)

Target zones (far side of table):
  red_area   @ (0.45,  0.25, 0.002)
  blue_area  @ (0.55,  0.25, 0.002)
  green_area @ (0.65,  0.25, 0.002)
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


# Canonical name ↔ asset key mapping used by Progress Manager / Policy.
CUBE_NAMES = ("red_cube", "blue_cube", "green_cube")
ZONE_NAMES = ("red_area", "blue_area", "green_area")

COLOR_RGB = {
    "red": (0.85, 0.15, 0.15),
    "blue": (0.15, 0.35, 0.90),
    "green": (0.15, 0.70, 0.30),
}

# Nucleus colored block USDs (same family as Isaac Lab stack task).
BLOCK_USD = {
    "red": f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
    "blue": f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
    "green": f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
}

CUBE_INIT_POS = {
    "red": (0.45, -0.20, 0.0203),
    "blue": (0.55, -0.20, 0.0203),
    "green": (0.65, -0.20, 0.0203),
}

ZONE_INIT_POS = {
    "red": (0.45, 0.25, 0.002),
    "blue": (0.55, 0.25, 0.002),
    "green": (0.65, 0.25, 0.002),
}


def _cube_rigid_props() -> RigidBodyPropertiesCfg:
    return RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        disable_gravity=False,
    )


def make_cube_cfg(color: str, prim_suffix: str) -> RigidObjectCfg:
    """Build a Franka-tabletop cube of the given color."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_suffix}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_INIT_POS[color], rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=BLOCK_USD[color],
            scale=(1.0, 1.0, 1.0),
            rigid_props=_cube_rigid_props(),
            semantic_tags=[("class", f"{color}_cube"), ("color", color)],
        ),
    )


def make_zone_cfg(color: str, prim_suffix: str) -> AssetBaseCfg:
    """
    Visual target zone (thin colored pad on the table).

    No rigid/collision props — zones are progress targets, not physical obstacles.
    """
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_suffix}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=ZONE_INIT_POS[color], rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.14, 0.14, 0.004),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=COLOR_RGB[color],
                metallic=0.05,
                roughness=0.4,
                opacity=0.85,
            ),
        ),
    )


@configclass
class ProgressFlowSceneCfg(InteractiveSceneCfg):
    """
    ProgressFlow interactive scene.

    Entity attributes are added in Isaac Lab's recommended order:
      terrain / ground → articulations / rigid bodies → sensors → lights.
    """

    # Robot (high-PD for absolute IK tracking)
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # End-effector frame sensor (filled below if not overridden)
    ee_frame: FrameTransformerCfg = MISSING

    # Cubes
    red_cube: RigidObjectCfg = make_cube_cfg("red", "RedCube")
    blue_cube: RigidObjectCfg = make_cube_cfg("blue", "BlueCube")
    green_cube: RigidObjectCfg = make_cube_cfg("green", "GreenCube")

    # Target zones (visual only)
    red_area: AssetBaseCfg = make_zone_cfg("red", "RedArea")
    blue_area: AssetBaseCfg = make_zone_cfg("blue", "BlueArea")
    green_area: AssetBaseCfg = make_zone_cfg("green", "GreenArea")

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.707, 0.0, 0.0, 0.707)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            semantic_tags=[("class", "table")],
        ),
    )

    # Ground
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


def build_ee_frame_cfg(debug_vis: bool = False) -> FrameTransformerCfg:
    """Franka panda_hand end-effector frame (with finger auxiliaries)."""
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"
    return FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=debug_vis,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.1034)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                name="tool_rightfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                name="tool_leftfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
        ],
    )


def finalize_scene_cfg(
    num_envs: int = 1,
    env_spacing: float = 2.5,
    debug_ee: bool = False,
) -> ProgressFlowSceneCfg:
    """Construct a ready-to-use scene cfg (called by env factory)."""
    scene = ProgressFlowSceneCfg(
        num_envs=num_envs,
        env_spacing=env_spacing,
        replicate_physics=False,
    )
    scene.ee_frame = build_ee_frame_cfg(debug_vis=debug_ee)
    scene.robot.spawn.semantic_tags = [("class", "robot")]
    return scene
