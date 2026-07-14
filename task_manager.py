"""Thin re-export so the repo root matches the planned layout."""

from progressflow.task_manager import (
    DEFAULT_INSTRUCTION,
    ParsedInstruction,
    TaskManager,
    build_default_task_manager,
)

__all__ = [
    "DEFAULT_INSTRUCTION",
    "ParsedInstruction",
    "TaskManager",
    "build_default_task_manager",
]


if __name__ == "__main__":
    tm = TaskManager()
    print(tm.describe())
