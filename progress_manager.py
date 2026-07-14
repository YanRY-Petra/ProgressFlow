"""Thin re-export so the repo root matches the planned layout."""

from progressflow.progress_manager import (
    ProgressManager,
    ProgressState,
    Subtask,
    SubtaskStatus,
)

__all__ = ["ProgressManager", "ProgressState", "Subtask", "SubtaskStatus"]


if __name__ == "__main__":
    from progressflow.task_manager import TaskManager

    tm = TaskManager()
    pm = tm.build_progress_manager()
    print(pm.get_state().summary())
    pm.mark_grasped()
    print(pm.get_state().summary())
    pm.mark_completed()
    print(pm.get_state().summary())
