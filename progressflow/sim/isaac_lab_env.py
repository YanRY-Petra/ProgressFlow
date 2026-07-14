"""
Isaac Lab adapter for ProgressFlow.

Wires:
  Language Instruction → Task Parser → Progress Manager → Robot Policy
    → PickPlaceController → Isaac Lab (Franka IK) → Progress / HUD / Eval

Scene assembly lives in ``progressflow.sim.isaac_lab.scene_cfg``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from progressflow.policy.policies import Action
from progressflow.progress_manager import ProgressManager
from progressflow.task_manager import TaskManager


@dataclass
class IsaacLabConfig:
    """Expected scene when plugging into Isaac Lab."""

    robot: str = "Franka Panda"
    table: bool = True
    cubes: tuple[str, ...] = ("red_cube", "blue_cube", "green_cube")
    zones: tuple[str, ...] = ("red_area", "blue_area", "green_area")
    control: str = "ik_abs"  # ik_abs | joint_pos
    env_id: str = "ProgressFlow-SequentialPickPlace-Franka-v0"
    num_envs: int = 1
    device: str = "cuda:0"
    headless: bool = False


class IsaacLabEnvAdapter:
    """
    Thin runtime wrapper around the ProgressFlow Isaac Lab environment.

    Usage (inside Isaac Sim / Isaac Lab python)::

        adapter = IsaacLabEnvAdapter()
        obs = adapter.reset()
        while not done:
            action = policy.act(obs, adapter.progress)
            obs, done = adapter.step(action)
    """

    def __init__(self, config: IsaacLabConfig | None = None, task_manager: TaskManager | None = None):
        self.config = config or IsaacLabConfig()
        self.task_manager = task_manager or TaskManager()
        self.progress: ProgressManager | None = None
        self._env = None
        self._controller = None
        self._step_count = 0
        self._active_high_level: Action | None = None
        self.wrong_picks = 0
        self.repeated_picks = 0
        self._placed_history: list[str] = []
        self.timeline: list[dict[str, Any]] = []

    def available(self) -> bool:
        try:
            import isaaclab  # noqa: F401
            import isaaclab_assets  # noqa: F401

            return True
        except Exception:
            return False

    def create_env(self) -> Any:
        """Assemble ProgressFlow scene + ManagerBasedRLEnv."""
        from isaaclab.envs import ManagerBasedRLEnv

        from progressflow.sim.isaac_lab.env_cfg import make_env_cfg

        cfg = make_env_cfg(
            num_envs=self.config.num_envs,
            control_mode=self.config.control,
            device=self.config.device,
        )
        return ManagerBasedRLEnv(cfg=cfg)

    def reset(self) -> dict[str, Any]:
        if not self.available():
            raise RuntimeError(
                "Isaac Lab is not installed. Use TableTopSim for local demos, "
                "or launch via:  ./isaaclab.sh -p scripts/run_isaac_lab_scene.py"
            )
        from progressflow.sim.isaac_lab.controller import PickPlaceController
        from progressflow.sim.isaac_lab.mdp import build_tabletop_observation

        if self._env is None:
            self._env = self.create_env()
        else:
            self._env.reset()

        self._controller = PickPlaceController(device=self.config.device)
        self.progress = self.task_manager.build_progress_manager()
        self._step_count = 0
        self.wrong_picks = 0
        self.repeated_picks = 0
        self._placed_history.clear()
        self.timeline.clear()
        self._active_high_level = None

        # Warm reset to spawn assets.
        self._env.reset()
        obs = build_tabletop_observation(self._env, env_id=0)
        obs["progress"] = self.progress.get_state().to_dict()
        obs["step"] = self._step_count
        return obs

    def step(self, action: Action) -> tuple[dict[str, Any], bool]:
        if self._env is None or self._controller is None or self.progress is None:
            raise RuntimeError("Call reset() first.")

        from progressflow.sim.isaac_lab.controller import (
            cube_positions_from_scene,
            pack_env_action,
        )
        from progressflow.sim.isaac_lab.mdp import build_tabletop_observation, object_in_zone

        # Track pick mistakes relative to current instructed object.
        desired = self.progress.current_task
        if action.kind in ("approach", "grasp", "pick") and action.object_name and desired:
            if action.object_name != desired.object_name:
                self.wrong_picks += 1
            if action.object_name in self._placed_history:
                self.repeated_picks += 1

        # (Re)start low-level primitive when high-level action changes.
        if self._active_high_level is None or self._action_key(action) != self._action_key(
            self._active_high_level
        ):
            self._controller.set_goal(action.kind, action.object_name, action.target_zone)
            self._active_high_level = action

        # Drive until primitive settles or safety cap.
        primitive_done = False
        for _ in range(200):
            cube_pos = cube_positions_from_scene(self._env.scene, self.config.device)
            origins = self._env.scene.env_origins
            ee = self._env.scene["ee_frame"].data.target_pos_w[:, 0, :]
            arm, grip, primitive_done = self._controller.compute(cube_pos, origins, ee)
            env_action = pack_env_action(arm, grip)
            self._env.step(env_action)
            self._step_count += 1
            if primitive_done or action.kind in ("idle", "observe"):
                break

        # Sync Progress Manager from physics.
        self._sync_progress(action, object_in_zone)

        obs = build_tabletop_observation(self._env, env_id=0)
        # Annotate grasp from controller phase.
        phase = self._controller.state.phase
        if phase in ("grasp_close", "lift", "transport", "place_lower", "place_open"):
            held = self._controller.state.object_name
            obs["grasped_object"] = held
            if held and held in obs["cubes"]:
                obs["cubes"][held]["grasped"] = True
        obs["progress"] = self.progress.get_state().to_dict()
        obs["step"] = self._step_count
        obs["controller_phase"] = phase

        self.timeline.append(
            {
                "step": self._step_count,
                "action": action.to_dict(),
                "progress": obs["progress"],
                "controller_phase": phase,
            }
        )

        done = self.progress.is_done or self._step_count >= 5000
        return obs, done

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    # ---- internals ----------------------------------------------------
    @staticmethod
    def _action_key(action: Action) -> tuple:
        return (action.kind, action.object_name, action.target_zone)

    def _sync_progress(self, action: Action, object_in_zone_fn) -> None:
        assert self.progress is not None and self._env is not None
        from progressflow.progress_manager import SubtaskStatus

        task = self.progress.current_task
        if task is None:
            return

        # Grasp bookkeeping
        if action.kind in ("grasp", "approach", "pick") and self._controller:
            if self._controller.state.phase in ("lift", "transport") and task.status == SubtaskStatus.ACTIVE:
                if self._controller.state.object_name == task.object_name:
                    self.progress.mark_grasped()

        if action.kind == "transport" and task.status == SubtaskStatus.ACTIVE:
            # transport implies already grasped in scripted demo
            if self._controller and self._controller.state.object_name == task.object_name:
                self.progress.mark_grasped()

        # Place success from physics proximity
        if action.kind in ("place", "release") or (
            self._controller and self._controller.state.phase in ("retreat", "idle")
        ):
            in_zone = bool(
                object_in_zone_fn(self._env, task.object_name, task.target_zone)[0].item()
            )
            if in_zone:
                if task.status == SubtaskStatus.ACTIVE:
                    self.progress.mark_grasped()
                if (
                    self.progress.current_task
                    and self.progress.current_task.status == SubtaskStatus.GRASPED
                    and self.progress.current_task.object_name == task.object_name
                ):
                    self.progress.mark_completed()
                    self._placed_history.append(task.object_name)


def recommend_assets() -> dict[str, str]:
    return {
        "robot": "Franka Emika Panda (FRANKA_PANDA_HIGH_PD_CFG)",
        "table": "SeattleLabTable/table_instanceable.usd",
        "objects": "red_block / blue_block / green_block.usd",
        "targets": "colored CuboidCfg pads (collision disabled)",
        "camera": "third-person viewport for recording",
    }


def describe_scene() -> str:
    from progressflow.sim.isaac_lab.scene_cfg import CUBE_INIT_POS, ZONE_INIT_POS

    lines = [
        "ProgressFlow Isaac Lab Scene",
        "  Robot : Franka Panda (high-PD, abs IK)",
        "  Table : SeattleLabTable",
        "  Cubes :",
    ]
    for color, pos in CUBE_INIT_POS.items():
        lines.append(f"    - {color}_cube @ {pos}")
    lines.append("  Zones :")
    for color, pos in ZONE_INIT_POS.items():
        lines.append(f"    - {color}_area @ {pos}")
    return "\n".join(lines)
