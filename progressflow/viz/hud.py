"""
ASCII / text visualization of instruction, progress bar, and current target.

Mirrors the on-screen HUD planned for the 1-minute demo video.
"""

from __future__ import annotations

from typing import Any

from ..progress_manager import ProgressState


def render_hud(state: ProgressState, width: int = 56) -> str:
    instr_lines = _wrap_instruction(state.instruction, width=width - 4)
    current = state.current_task.object_name if state.current_task else ("DONE" if state.progress_value >= 1 else "None")
    completed = ", ".join(t.object_color for t in state.completed_tasks) or "-"
    remaining = ", ".join(t.object_color for t in state.remaining_tasks) or "-"

    bar = _rich_bar(state.progress_value, 20)
    lines = [
        "┌" + "─" * (width - 2) + "┐",
        _row("INSTRUCTION", width),
        *[f"│ {line:<{width - 4}} │" for line in instr_lines],
        "├" + "─" * (width - 2) + "┤",
        _row(f"PROGRESS  {state.progress_bar}  {state.progress_percent}%", width),
        _row(f"{bar}", width),
        _row(f"Phase: {state.phase:<12}  Progress Value: {state.progress_value:.2f}", width),
        "├" + "─" * (width - 2) + "┤",
        _row(f"Current Target : {current}", width),
        _row(f"Completed      : {completed}", width),
        _row(f"Remaining      : {remaining}", width),
        "└" + "─" * (width - 2) + "┘",
    ]
    return "\n".join(lines)


def render_scene_snapshot(obs: dict[str, Any]) -> str:
    cubes = obs.get("cubes", {})
    rows = ["Tabletop snapshot:"]
    for name, data in cubes.items():
        zone = data.get("zone") or "on_table"
        flag = "LOCKED" if data.get("grasped") else zone
        rows.append(f"  • {name:<12} @ {data.get('position')}  [{flag}]")
    grasped = obs.get("grasped_object")
    rows.append(f"  gripper: {grasped or 'empty'}")
    return "\n".join(rows)


def render_demo_frame(obs: dict[str, Any], action_kind: str | None = None) -> str:
    progress = obs.get("progress")
    if not progress:
        return render_scene_snapshot(obs)
    # Rebuild a lightweight ProgressState-like view from dict for HUD.
    from ..progress_manager import ProgressState, Subtask, SubtaskStatus

    def _maybe(d: dict[str, Any] | None) -> Subtask | None:
        if not d:
            return None
        return Subtask(
            task_id=d["task_id"],
            object_name=d["object_name"],
            object_color=d["object_color"],
            target_zone=d["target_zone"],
            status=SubtaskStatus(d["status"]),
            attempts=d.get("attempts", 0),
        )

    state = ProgressState(
        current_task=_maybe(progress.get("current_task")),
        completed_tasks=[_maybe(t) for t in progress.get("completed_tasks", []) if t],  # type: ignore[misc]
        remaining_tasks=[_maybe(t) for t in progress.get("remaining_tasks", []) if t],  # type: ignore[misc]
        progress_value=progress["progress_value"],
        phase=progress["phase"],
        total_tasks=progress["total_tasks"],
        instruction=progress.get("instruction", ""),
    )
    parts = [render_hud(state), "", render_scene_snapshot(obs)]
    if action_kind:
        parts.append(f"\nAction: {action_kind}")
    return "\n".join(parts)


def _row(text: str, width: int) -> str:
    return f"│ {text:<{width - 4}} │"


def _wrap_instruction(text: str, width: int) -> list[str]:
    if not text:
        return [""]
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        trial = (" ".join(cur + [w])).strip()
        if len(trial) <= width:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines or [""]


def _rich_bar(value: float, width: int) -> str:
    filled = max(0, min(width, int(round(value * width))))
    return "[" + "#" * filled + "-" * (width - filled) + "]"
