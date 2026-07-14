"""
Task Manager — parse language instruction into ordered subtasks.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Iterable

from .progress_manager import ProgressManager, Subtask


DEFAULT_INSTRUCTION = (
    "Move the red cube to the red box. "
    "Then move the blue cube to the blue box. "
    "Finally move the green cube to the green box."
)

COLOR_ALIASES = {
    "red": "red",
    "blue": "blue",
    "green": "green",
}


@dataclass
class ParsedInstruction:
    raw: str
    subtasks: list[Subtask]

    def summary_lines(self) -> list[str]:
        return [
            f"{i + 1}. Move {t.object_color} cube → {t.target_zone}"
            for i, t in enumerate(self.subtasks)
        ]


class TaskManager:
    """
    Converts a natural-language sequential instruction into a ProgressManager.

    Supports patterns like:
      "Move the red cube to the red box. Then move the blue cube ..."
    """

    SENTENCE_SPLIT = re.compile(r"[.!?]+|\b(?:then|finally|next|after that)\b", re.I)
    MOVE_PATTERN = re.compile(
        r"move\s+the\s+(?P<obj_color>red|blue|green)\s+cube\s+to\s+the\s+"
        r"(?P<tgt_color>red|blue|green)\s+(?:box|area|zone)",
        re.I,
    )

    def __init__(self, instruction: str = DEFAULT_INSTRUCTION):
        self.instruction = " ".join(instruction.split())
        self.parsed = self.parse(self.instruction)

    @classmethod
    def parse(cls, instruction: str) -> ParsedInstruction:
        text = " ".join(instruction.split())
        subtasks: list[Subtask] = []
        # Prefer explicit move-clause matching over brittle sentence splits.
        for match in cls.MOVE_PATTERN.finditer(text):
            obj_color = COLOR_ALIASES[match.group("obj_color").lower()]
            tgt_color = COLOR_ALIASES[match.group("tgt_color").lower()]
            subtasks.append(
                Subtask(
                    task_id=len(subtasks) + 1,
                    object_name=f"{obj_color}_cube",
                    object_color=obj_color,
                    target_zone=f"{tgt_color}_area",
                )
            )
        if not subtasks:
            raise ValueError(
                "Could not parse any pick-and-place subtasks from instruction:\n"
                f"  {instruction}"
            )
        return ParsedInstruction(raw=text, subtasks=subtasks)

    def build_progress_manager(self) -> ProgressManager:
        # Deep-copy so repeated episodes do not share mutable Subtask status.
        return ProgressManager.from_instruction(
            self.instruction, copy.deepcopy(self.parsed.subtasks)
        )

    @staticmethod
    def default_cubes() -> list[str]:
        return ["red_cube", "blue_cube", "green_cube"]

    @staticmethod
    def default_zones() -> list[str]:
        return ["red_area", "blue_area", "green_area"]

    def describe(self) -> str:
        lines = ["Instruction parsed into subtasks:"]
        lines.extend(f"  {line}" for line in self.parsed.summary_lines())
        return "\n".join(lines)


def build_default_task_manager(instruction: str | None = None) -> TaskManager:
    return TaskManager(instruction or DEFAULT_INSTRUCTION)


def ensure_subtask_order(subtasks: Iterable[Subtask]) -> list[Subtask]:
    ordered = list(subtasks)
    for i, t in enumerate(ordered, start=1):
        t.task_id = i
    return ordered
