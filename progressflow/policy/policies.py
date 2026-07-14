"""
Robot policies for sequential pick-and-place.

Two variants for the evaluation story:
  - ProgressAwarePolicy: consults ProgressManager before acting
  - BaselinePolicy: myopic; no progress memory → wrong/repeated picks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..progress_manager import ProgressManager


@dataclass
class Action:
    kind: str  # observe | approach | grasp | transport | place | release | idle
    object_name: str | None = None
    target_zone: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "object_name": self.object_name,
            "target_zone": self.target_zone,
            "meta": self.meta,
        }


class Policy(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(self, observation: dict[str, Any], progress: ProgressManager | None) -> Action: ...


@dataclass
class ProgressAwarePolicy:
    """
    Uses ProgressManager as working memory.

    Observe → Pick → Transport → Place → Update Progress → Next Task
    """

    name: str = "progress_aware"

    def reset(self) -> None:
        return

    def act(self, observation: dict[str, Any], progress: ProgressManager | None) -> Action:
        if progress is None or progress.is_done:
            return Action(kind="idle")

        task = progress.current_task
        if task is None:
            return Action(kind="idle")

        phase = progress.phase
        obj = task.object_name
        zone = task.target_zone

        if phase in ("observe", "update"):
            progress.set_phase("pick")
            return Action(kind="approach", object_name=obj, target_zone=zone)

        if phase == "pick":
            grasped = observation.get("grasped_object")
            if grasped == obj:
                progress.mark_grasped()
                return Action(kind="grasp", object_name=obj, target_zone=zone)
            return Action(kind="grasp", object_name=obj, target_zone=zone)

        if phase == "transport":
            progress.set_phase("place")
            return Action(kind="transport", object_name=obj, target_zone=zone)

        if phase == "place":
            placed = observation.get("object_in_zone", {}).get(obj) == zone
            if placed or observation.get("last_placed") == obj:
                # Progress is usually advanced by the simulator on a correct place;
                # keep this guard for observability-only loops.
                if progress.current_task and progress.current_task.object_name == obj:
                    if progress.current_task.status.value == "grasped":
                        progress.mark_completed()
                return Action(kind="place", object_name=obj, target_zone=zone)
            return Action(kind="place", object_name=obj, target_zone=zone)

        if phase == "done":
            return Action(kind="idle")

        return Action(kind="observe", object_name=obj, target_zone=zone)


@dataclass
class BaselinePolicy:
    """
    No progress state. Heuristics over raw observation only.

    Typical failure modes for long-horizon demos:
      - repeat pick on already-placed cubes
      - forget the next required cube / stall
      - wrong color association under partial success
    """

    name: str = "baseline"
    confusion_rate: float = 0.55
    forget_rate: float = 0.25
    _rng_seed: int = 0
    _step: int = 0
    _mode: str = "hunt"  # hunt | holding
    _holding: str | None = None
    _holding_zone: str | None = None

    def reset(self) -> None:
        self._step = 0
        self._mode = "hunt"
        self._holding = None
        self._holding_zone = None

    def act(self, observation: dict[str, Any], progress: ProgressManager | None) -> Action:
        # Intentionally ignore `progress` even if provided.
        del progress
        self._step += 1
        cubes = observation.get("cubes", {})
        order = observation.get("instruction_order", list(cubes.keys()))

        unplaced = [name for name in order if cubes.get(name, {}).get("zone") is None]
        placed = [name for name in order if cubes.get(name, {}).get("zone") is not None]

        if not unplaced and self._mode != "holding":
            return Action(kind="idle")

        # Holding → maybe place wrong / drop / finally place
        if self._mode == "holding" and self._holding:
            obj = self._holding
            zone = self._holding_zone or obj.replace("_cube", "_area")
            r = self._pseudo_rand()
            if r < self.forget_rate:
                # Forget what we are doing: re-grasp something else.
                self._mode = "hunt"
                distractor = (placed + unplaced + [obj])[self._step % max(len(placed + unplaced + [obj]), 1)]
                return Action(
                    kind="grasp",
                    object_name=distractor,
                    target_zone=distractor.replace("_cube", "_area"),
                    meta={"bug": "forgot_next"},
                )
            if r < self.forget_rate + self.confusion_rate * 0.35 and placed:
                # Place onto a wrong zone associated with an already finished cube.
                wrong = placed[self._step % len(placed)]
                wrong_zone = wrong.replace("_cube", "_area")
                self._mode = "hunt"
                self._holding = None
                return Action(
                    kind="place",
                    object_name=obj,
                    target_zone=wrong_zone,
                    meta={"bug": "wrong_place"},
                )
            self._mode = "hunt"
            self._holding = None
            self._holding_zone = None
            return Action(kind="place", object_name=obj, target_zone=zone)

        # Memoryless hunt: often re-pick finished cubes or pick the wrong remaining one.
        r = self._pseudo_rand()
        if placed and r < self.confusion_rate:
            target = placed[self._step % len(placed)]
            return Action(
                kind="grasp",
                object_name=target,
                target_zone=target.replace("_cube", "_area"),
                meta={"bug": "repeated_pick"},
            )

        if len(unplaced) > 1 and r < self.confusion_rate + self.forget_rate * 0.5:
            # Skip the "next" cube — classic long-horizon memory failure.
            target = unplaced[-1]
            wrong_zone = unplaced[0].replace("_cube", "_area")
            self._mode = "holding"
            self._holding = target
            self._holding_zone = wrong_zone
            return Action(
                kind="grasp",
                object_name=target,
                target_zone=wrong_zone,
                meta={"bug": "wrong_pick"},
            )

        if not unplaced:
            return Action(kind="idle")

        target = unplaced[0]
        zone = target.replace("_cube", "_area")
        grasped = observation.get("grasped_object")
        if grasped == target:
            self._mode = "holding"
            self._holding = target
            self._holding_zone = zone
            return Action(kind="transport", object_name=target, target_zone=zone)

        if self._step % 2 == 1:
            return Action(kind="approach", object_name=target, target_zone=zone)
        self._mode = "holding"
        self._holding = target
        self._holding_zone = zone
        return Action(kind="grasp", object_name=target, target_zone=zone)

    def _pseudo_rand(self) -> float:
        self._rng_seed = (1103515245 * (self._rng_seed + self._step + 1) + 12345) % (2**31)
        return (self._rng_seed % 10000) / 10000.0
