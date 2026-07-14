"""
Lightweight tabletop simulator for ProgressFlow demos/evaluation.

Designed so the Progress Manager + Policy loop runs without Isaac Lab.
An Isaac Lab adapter (``isaac_lab_env.py``) can wrap the same Observation/Action API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..policy.policies import Action
from ..progress_manager import ProgressManager
from ..task_manager import TaskManager


COLOR_RGB = {
    "red": (0.85, 0.18, 0.18),
    "blue": (0.18, 0.40, 0.90),
    "green": (0.18, 0.72, 0.35),
}


@dataclass
class CubeState:
    name: str
    color: str
    position: tuple[float, float, float]
    zone: str | None = None
    grasped: bool = False


@dataclass
class ZoneState:
    name: str
    color: str
    position: tuple[float, float, float]


@dataclass
class SimConfig:
    max_steps: int = 60
    grasp_success_prob: float = 1.0
    place_success_prob: float = 1.0
    seed: int = 0


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    progress_final: float
    wrong_picks: int
    repeated_picks: int
    completed_subtasks: int
    total_subtasks: int
    timeline: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "steps": self.steps,
            "progress_final": self.progress_final,
            "wrong_picks": self.wrong_picks,
            "repeated_picks": self.repeated_picks,
            "completed_subtasks": self.completed_subtasks,
            "total_subtasks": self.total_subtasks,
            "average_completion": self.completed_subtasks / max(self.total_subtasks, 1),
        }


class TableTopSim:
    """
    Scene:
      Franka Panda (logical) · Table · 3 Cubes · 3 Target Zones
    """

    def __init__(self, task_manager: TaskManager, config: SimConfig | None = None):
        self.task_manager = task_manager
        self.config = config or SimConfig()
        self.progress: ProgressManager | None = None
        self.cubes: dict[str, CubeState] = {}
        self.zones: dict[str, ZoneState] = {}
        self.grasped: str | None = None
        self.step_count = 0
        self.wrong_picks = 0
        self.repeated_picks = 0
        self._placed_history: list[str] = []
        self._pick_history: list[str] = []
        self.timeline: list[dict[str, Any]] = []
        self._rng = self.config.seed
        self.reset()

    def reset(self) -> dict[str, Any]:
        self.progress = self.task_manager.build_progress_manager()
        self.step_count = 0
        self.wrong_picks = 0
        self.repeated_picks = 0
        self._placed_history.clear()
        self._pick_history.clear()
        self.timeline.clear()
        self.grasped = None

        # Layout: cubes on the near side of the table, color zones farther.
        cube_x = [0.45, 0.55, 0.65]
        zone_x = [0.45, 0.55, 0.65]
        colors = ["red", "blue", "green"]
        self.cubes = {
            f"{c}_cube": CubeState(
                name=f"{c}_cube",
                color=c,
                position=(cube_x[i], -0.20, 0.05),
            )
            for i, c in enumerate(colors)
        }
        self.zones = {
            f"{c}_area": ZoneState(
                name=f"{c}_area",
                color=c,
                position=(zone_x[i], 0.25, 0.01),
            )
            for i, c in enumerate(colors)
        }
        return self.get_observation()

    def get_observation(self) -> dict[str, Any]:
        assert self.progress is not None
        object_in_zone = {n: c.zone for n, c in self.cubes.items()}
        return {
            "cubes": {
                n: {
                    "color": c.color,
                    "position": c.position,
                    "zone": c.zone,
                    "grasped": c.grasped,
                }
                for n, c in self.cubes.items()
            },
            "zones": {
                n: {"color": z.color, "position": z.position} for n, z in self.zones.items()
            },
            "grasped_object": self.grasped,
            "object_in_zone": object_in_zone,
            "last_placed": self._placed_history[-1] if self._placed_history else None,
            "instruction_order": [t.object_name for t in self.progress.subtasks],
            "progress": self.progress.get_state().to_dict(),
            "step": self.step_count,
        }

    def step(self, action: Action) -> tuple[dict[str, Any], bool]:
        assert self.progress is not None
        self.step_count += 1
        desired = self.progress.current_task

        if action.kind in ("approach", "grasp") and action.object_name:
            self._register_pick(action.object_name, desired.object_name if desired else None)
            if action.kind == "grasp":
                self._do_grasp(action.object_name)

        if action.kind == "transport" and self.grasped:
            # Move cube toward target zone (logical).
            zone = self.zones.get(action.target_zone or "")
            cube = self.cubes[self.grasped]
            if zone:
                cube.position = (zone.position[0], zone.position[1], 0.12)

        if action.kind in ("place", "release") and action.object_name:
            self._do_place(action.object_name, action.target_zone)

        obs = self.get_observation()
        state = self.progress.get_state()
        self.timeline.append(
            {
                "step": self.step_count,
                "action": action.to_dict(),
                "progress": state.to_dict(),
            }
        )
        done = self.progress.is_done or self.step_count >= self.config.max_steps
        return obs, done

    def run_episode(self, policy, use_progress: bool = True) -> EpisodeResult:
        self.reset()
        policy.reset()
        done = False
        obs = self.get_observation()

        while not done:
            progress = self.progress if use_progress else None
            # Progress-aware policy needs live manager; baseline ignores it.
            if use_progress:
                action = policy.act(obs, self.progress)
            else:
                action = policy.act(obs, None)
                # Sync a shadow progress for metrics/visualization only when baseline
                # actually places the currently expected object correctly.
                self._shadow_update_for_baseline(action, obs)

            obs, done = self.step(action)

        assert self.progress is not None
        completed = len(self.progress.completed_tasks)
        total = len(self.progress.subtasks)
        physical_ok = all(
            self.cubes[t.object_name].zone == t.target_zone for t in self.progress.subtasks
        )
        return EpisodeResult(
            success=self.progress.is_done and physical_ok,
            steps=self.step_count,
            progress_final=self.progress.progress_value,
            wrong_picks=self.wrong_picks,
            repeated_picks=self.repeated_picks,
            completed_subtasks=completed if physical_ok else sum(
                1
                for t in self.progress.subtasks
                if self.cubes[t.object_name].zone == t.target_zone
            ),
            total_subtasks=total,
            timeline=list(self.timeline),
        )

    # ---- internals ----------------------------------------------------
    def _rand(self) -> float:
        self._rng = (1103515245 * self._rng + 12345) % (2**31)
        return (self._rng % 1000) / 1000.0

    def _register_pick(self, object_name: str, desired: str | None) -> None:
        if desired and object_name != desired:
            self.wrong_picks += 1
        if object_name in self._placed_history:
            self.repeated_picks += 1
        if self._pick_history.count(object_name) > 0 and object_name in self._placed_history:
            self.repeated_picks += 1
        self._pick_history.append(object_name)

    def _do_grasp(self, object_name: str) -> None:
        if object_name not in self.cubes:
            return
        if self._rand() > self.config.grasp_success_prob:
            return
        # Release previous if any.
        if self.grasped and self.grasped in self.cubes:
            self.cubes[self.grasped].grasped = False
        for c in self.cubes.values():
            c.grasped = False
        self.cubes[object_name].grasped = True
        self.cubes[object_name].zone = None
        self.grasped = object_name

    def _do_place(self, object_name: str, target_zone: str | None) -> None:
        if object_name not in self.cubes:
            return
        if self.grasped != object_name:
            # Allow place if already grasped earlier in same step cycle.
            if not self.cubes[object_name].grasped:
                return
        if self._rand() > self.config.place_success_prob:
            return
        zone_name = target_zone or object_name.replace("_cube", "_area")
        if not isinstance(zone_name, str):
            return
        zone = self.zones.get(zone_name)
        cube = self.cubes[object_name]
        if zone is None:
            return
        cube.position = (zone.position[0], zone.position[1], 0.05)
        cube.zone = zone_name
        cube.grasped = False
        self.grasped = None
        self._placed_history.append(object_name)

        # Advance ProgressManager only on the currently instructed correct place.
        assert self.progress is not None
        from ..progress_manager import SubtaskStatus

        current = self.progress.current_task
        if not current:
            return
        if current.object_name != object_name or current.target_zone != zone_name:
            return
        if current.status == SubtaskStatus.ACTIVE:
            self.progress.mark_grasped()
        if (
            self.progress.current_task
            and self.progress.current_task.status == SubtaskStatus.GRASPED
            and self.progress.current_task.object_name == object_name
        ):
            self.progress.mark_completed()

    def _shadow_update_for_baseline(self, action: Action, obs: dict[str, Any]) -> None:
        """No-op hook: progress is updated inside ``_do_place`` when correct."""
        return
