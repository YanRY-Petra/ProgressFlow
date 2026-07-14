"""
Manager-based Isaac Lab environment configuration for ProgressFlow.

Control mode defaults to absolute differential IK (Franka high-PD), matching
``configs/isaac_lab.yaml``.
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab.envs.mdp as isaac_mdp

from . import mdp as pf_mdp
from .scene_cfg import ProgressFlowSceneCfg, finalize_scene_cfg


@configclass
class ActionsCfg:
    """Arm + gripper. Arm term is filled in ``__post_init__`` by control mode."""

    arm_action: JointPositionActionCfg | DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: BinaryJointPositionActionCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observations for debugging / optional learned policies."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel)
        actions = ObsTerm(func=isaac_mdp.last_action)
        ee_pos = ObsTerm(func=pf_mdp.ee_pos_w)
        red_cube_pos = ObsTerm(
            func=pf_mdp.cube_root_pos_w, params={"asset_cfg": SceneEntityCfg("red_cube")}
        )
        blue_cube_pos = ObsTerm(
            func=pf_mdp.cube_root_pos_w, params={"asset_cfg": SceneEntityCfg("blue_cube")}
        )
        green_cube_pos = ObsTerm(
            func=pf_mdp.cube_root_pos_w, params={"asset_cfg": SceneEntityCfg("green_cube")}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset whole scene to default poses."""

    reset_all = EventTerm(func=isaac_mdp.reset_scene_to_default, mode="reset")


@configclass
class TerminationsCfg:
    """Episode ends on timeout or full color-sort success."""

    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)
    success = DoneTerm(func=pf_mdp.all_cubes_sorted)


@configclass
class ProgressFlowEnvCfg(ManagerBasedRLEnvCfg):
    """
    ProgressFlow long-horizon sequential pick-and-place environment.

    Rewards are unused (scripted / progress-aware controller drives the demo).
    """

    scene: ProgressFlowSceneCfg = ProgressFlowSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    commands = None
    rewards = None
    curriculum = None

    # Extra knobs consumed by the runner / adapter (not Isaac Lab managers).
    control_mode: str = "ik_abs"  # ik_abs | joint_pos
    zone_xy_thresh: float = 0.07

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 90.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # Assemble Franka + table + cubes + zones + EE frame.
        self.scene = finalize_scene_cfg(
            num_envs=self.scene.num_envs,
            env_spacing=self.scene.env_spacing,
            debug_ee=False,
        )

        if self.control_mode == "joint_pos":
            self.actions.arm_action = JointPositionActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                scale=0.5,
                use_default_offset=True,
            )
        else:
            # Absolute EE pose IK (recommended for demos).
            self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                body_name="panda_hand",
                controller=DifferentialIKControllerCfg(
                    command_type="pose",
                    use_relative_mode=False,
                    ik_method="dls",
                ),
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
            )


@configclass
class ProgressFlowEnvCfg_PLAY(ProgressFlowEnvCfg):
    """Play / recording cfg (defaults to 1 env; override via ``make_env_cfg``)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.episode_length_s = 120.0


def make_env_cfg(
    num_envs: int = 1,
    control_mode: str = "ik_abs",
    device: str = "cuda:0",
) -> ProgressFlowEnvCfg:
    """Factory used by scripts and ``IsaacLabEnvAdapter``."""
    cfg = ProgressFlowEnvCfg_PLAY()
    cfg.control_mode = control_mode
    cfg.scene.num_envs = num_envs
    # Rebuild scene / action wiring with the requested control mode & env count.
    cfg.__post_init__()
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    return cfg
