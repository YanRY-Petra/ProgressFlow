"""
Progress Manager — the core of ProgressFlow.

Inspired by PALM's progress-aware formulation: instead of treating a long-horizon
instruction as a single opaque goal, we expose an explicit progress state that
the policy and visualization can consume.

This implementation is intentionally rule-based (no neural network). Future work
can replace it with learned progress prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SubtaskStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    GRASPED = "grasped"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Subtask:
    """One pick-and-place unit in the long-horizon instruction."""

    task_id: int
    object_name: str  # e.g. "red_cube"
    object_color: str  # e.g. "red"
    target_zone: str  # e.g. "red_area"
    status: SubtaskStatus = SubtaskStatus.PENDING
    attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "object_name": self.object_name,
            "object_color": self.object_color,
            "target_zone": self.target_zone,
            "status": self.status.value,
            "attempts": self.attempts,
        }


@dataclass
class ProgressState:
    """Snapshot of long-horizon progress, analogous to PALM progress values."""

    current_task: Subtask | None
    completed_tasks: list[Subtask]
    remaining_tasks: list[Subtask]
    progress_value: float  # in [0, 1]
    phase: str  # observe | pick | transport | place | update | done
    total_tasks: int
    instruction: str = ""

    @property
    def progress_percent(self) -> int:
        return int(round(self.progress_value * 100))

    @property
    def progress_bar(self) -> str:
        filled = round(self.progress_value * 3)
        return "█" * filled + "░" * (3 - filled)

    def summary(self) -> str:
        current = self.current_task.object_name if self.current_task else "None"
        completed = [t.object_name for t in self.completed_tasks]
        remaining = [t.object_name for t in self.remaining_tasks]
        return (
            f"Progress {self.progress_bar} {self.progress_percent}% | "
            f"Current: {current} | Phase: {self.phase} | "
            f"Completed: {completed} | Remaining: {remaining}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_task": self.current_task.to_dict() if self.current_task else None,
            "completed_tasks": [t.to_dict() for t in self.completed_tasks],
            "remaining_tasks": [t.to_dict() for t in self.remaining_tasks],
            "progress_value": self.progress_value,
            "progress_percent": self.progress_percent,
            "progress_bar": self.progress_bar,
            "phase": self.phase,
            "total_tasks": self.total_tasks,
            "instruction": self.instruction,
        }


@dataclass
class ProgressManager:
    """
    Tracks multi-subtask progress for sequential manipulation.

    Example timeline
    ----------------
    Task 1 red cube   status = pending
    grasp red         status = grasped
    place red         status = completed  → Progress 1/3
    Task 2 blue cube  ...
    finally           Progress 3/3
    """

    subtasks: list[Subtask] = field(default_factory=list)
    phase: str = "observe"
    instruction: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)

    # ---- construction -------------------------------------------------
    @classmethod
    def from_instruction(cls, instruction: str, subtasks: list[Subtask]) -> "ProgressManager":
        mgr = cls(subtasks=list(subtasks), instruction=instruction)
        if mgr.subtasks:
            mgr.subtasks[0].status = SubtaskStatus.ACTIVE
        mgr._log("init")
        return mgr

    # ---- queries ------------------------------------------------------
    @property
    def current_index(self) -> int:
        for i, t in enumerate(self.subtasks):
            if t.status in (SubtaskStatus.ACTIVE, SubtaskStatus.GRASPED):
                return i
        # if all done or none active, point past the last completed
        completed = sum(1 for t in self.subtasks if t.status == SubtaskStatus.COMPLETED)
        return min(completed, max(len(self.subtasks) - 1, 0))

    @property
    def current_task(self) -> Subtask | None:
        for t in self.subtasks:
            if t.status in (SubtaskStatus.ACTIVE, SubtaskStatus.GRASPED):
                return t
        return None

    @property
    def completed_tasks(self) -> list[Subtask]:
        return [t for t in self.subtasks if t.status == SubtaskStatus.COMPLETED]

    @property
    def remaining_tasks(self) -> list[Subtask]:
        return [
            t
            for t in self.subtasks
            if t.status in (SubtaskStatus.PENDING, SubtaskStatus.ACTIVE, SubtaskStatus.GRASPED)
        ]

    @property
    def progress_value(self) -> float:
        if not self.subtasks:
            return 0.0
        done = sum(1 for t in self.subtasks if t.status == SubtaskStatus.COMPLETED)
        # fine-grained: grasping current counts as half a step
        partial = 0.5 if self.current_task and self.current_task.status == SubtaskStatus.GRASPED else 0.0
        return min(1.0, (done + partial) / len(self.subtasks))

    @property
    def is_done(self) -> bool:
        return bool(self.subtasks) and all(
            t.status == SubtaskStatus.COMPLETED for t in self.subtasks
        )

    def get_state(self) -> ProgressState:
        return ProgressState(
            current_task=self.current_task,
            completed_tasks=self.completed_tasks,
            remaining_tasks=[
                t for t in self.subtasks if t.status == SubtaskStatus.PENDING
            ],
            progress_value=self.progress_value,
            phase=self.phase,
            total_tasks=len(self.subtasks),
            instruction=self.instruction,
        )

    # ---- updates ------------------------------------------------------
    def set_phase(self, phase: str) -> ProgressState:
        self.phase = phase
        self._log("phase")
        return self.get_state()

    def mark_grasped(self) -> ProgressState:
        task = self.current_task
        if task is None:
            raise RuntimeError("No active task to mark as grasped.")
        task.status = SubtaskStatus.GRASPED
        task.attempts += 1
        self.phase = "transport"
        self._log("grasped")
        return self.get_state()

    def mark_completed(self) -> ProgressState:
        task = self.current_task
        if task is None:
            raise RuntimeError("No active task to mark as completed.")
        task.status = SubtaskStatus.COMPLETED
        self.phase = "update"
        self._log("completed")
        self._activate_next()
        return self.get_state()

    def mark_failed(self) -> ProgressState:
        task = self.current_task
        if task is None:
            raise RuntimeError("No active task to mark as failed.")
        task.attempts += 1
        # stay on same object; reset to active for retry
        task.status = SubtaskStatus.ACTIVE
        self.phase = "observe"
        self._log("failed")
        return self.get_state()

    def _activate_next(self) -> None:
        for t in self.subtasks:
            if t.status == SubtaskStatus.PENDING:
                t.status = SubtaskStatus.ACTIVE
                self.phase = "observe"
                return
        self.phase = "done"

    def _log(self, event: str) -> None:
        state = self.get_state()
        self.history.append({"event": event, **state.to_dict()})


# Backward-friendly module-level alias matching the planned repo layout.
progress_manager = ProgressManager
