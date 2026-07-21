"""
Robot policies for sequential pick-and-place.

Fair ablation under shared decision noise:
  - BaselinePolicy: prompt plan + world state (no ProgressManager)
  - ProgressAwarePolicy: consults ProgressManager every step
  - DecisionNoiseWrapper: same confusion/forget noise applied on top of either
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


def _prompt_plan(observation: dict[str, Any]) -> list[dict[str, str]]:
    plan = observation.get("instruction_plan")
    if plan:
        return plan
    order = observation.get("instruction_order", list(observation.get("cubes", {}).keys()))
    return [
        {"object_name": name, "target_zone": name.replace("_cube", "_area")}
        for name in order
    ]


def _next_from_prompt(observation: dict[str, Any]) -> tuple[str, str] | None:
    """Next unfinished subtask from the static prompt + world state."""
    cubes = observation.get("cubes", {})
    for item in _prompt_plan(observation):
        obj = item["object_name"]
        zone = item["target_zone"]
        if cubes.get(obj, {}).get("zone") != zone:
            return obj, zone
    return None


@dataclass
class ProgressAwarePolicy:
    """
    Uses ProgressManager as working memory.

    Observe → Pick → Transport → Place → Update Progress → Next Task.
    Re-syncs phase from observation so it can recover after decision noise.
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

        obj = task.object_name
        zone = task.target_zone
        grasped = observation.get("grasped_object")

        # Recover from decision noise: if we should be transporting/placing but
        # are not holding the instructed object, go back to pick.
        if progress.phase in ("transport", "place") and grasped != obj:
            progress.set_phase("pick")

        phase = progress.phase

        if phase in ("observe", "update"):
            progress.set_phase("pick")
            return Action(kind="approach", object_name=obj, target_zone=zone)

        if phase == "pick":
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
    Prompt-only policy: static instruction plan + world state.
    Does not read ProgressManager. Decision noise is applied externally
    via DecisionNoiseWrapper (same as progress-aware).
    """

    name: str = "baseline"
    _step: int = 0
    _phase: str = "approach"  # approach | grasp | transport | place
    _obj: str | None = None
    _zone: str | None = None

    def reset(self) -> None:
        self._step = 0
        self._phase = "approach"
        self._obj = None
        self._zone = None

    def act(self, observation: dict[str, Any], progress: ProgressManager | None) -> Action:
        del progress
        self._step += 1

        nxt = _next_from_prompt(observation)
        if nxt is None:
            return Action(kind="idle")

        obj, zone = nxt
        if (obj, zone) != (self._obj, self._zone):
            self._obj = obj
            self._zone = zone
            self._phase = "approach"

        grasped = observation.get("grasped_object")
        # Recover local phase if noise left us without the intended object.
        if self._phase in ("transport", "place") and grasped != obj:
            self._phase = "grasp"

        if self._phase == "approach":
            self._phase = "grasp"
            return Action(kind="approach", object_name=obj, target_zone=zone, meta={"source": "prompt"})

        if self._phase == "grasp":
            if grasped == obj:
                self._phase = "transport"
            return Action(kind="grasp", object_name=obj, target_zone=zone, meta={"source": "prompt"})

        if self._phase == "transport":
            self._phase = "place"
            return Action(kind="transport", object_name=obj, target_zone=zone, meta={"source": "prompt"})

        if observation.get("object_in_zone", {}).get(obj) == zone or observation.get("last_placed") == obj:
            self._phase = "approach"
            self._obj = None
            self._zone = None
        return Action(kind="place", object_name=obj, target_zone=zone, meta={"source": "prompt"})


@dataclass
class DecisionNoiseWrapper:
    """
    Shared decision-side disturbance applied on top of any base policy.

    Models memoryless / forgetful mistakes under uncertainty:
      - confusion: swap grasp/approach target (often to an already-placed cube)
      - forget: abandon a place and re-grasp something else, or place to a wrong zone

    Same rates/seeds for baseline and progress-aware → fair ablation.
    """

    base: Any
    confusion_rate: float = 0.55
    forget_rate: float = 0.30
    _rng_seed: int = 0
    _step: int = 0

    @property
    def name(self) -> str:
        return getattr(self.base, "name", "noisy")

    def reset(self) -> None:
        self.base.reset()
        self._step = 0

    def act(self, observation: dict[str, Any], progress: ProgressManager | None) -> Action:
        action = self.base.act(observation, progress)
        return self._corrupt(action, observation)

    def _corrupt(self, action: Action, observation: dict[str, Any]) -> Action:
        if action.kind == "idle" or not action.object_name:
            return action

        # Only disturb high-level grasp/place decisions (not every approach/transport).
        if action.kind not in ("grasp", "place", "release"):
            return action

        self._step += 1
        cubes = observation.get("cubes", {})
        order = observation.get("instruction_order", list(cubes.keys()))
        if not order:
            return action

        unplaced = [n for n in order if cubes.get(n, {}).get("zone") is None]
        placed = [n for n in order if cubes.get(n, {}).get("zone") is not None]
        r = self._pseudo_rand()
        meta = dict(action.meta)

        if action.kind == "grasp":
            if placed and r < self.confusion_rate:
                wrong = placed[self._step % len(placed)]
                meta["noise"] = "confusion_repeated_pick"
                return Action(
                    kind="grasp",
                    object_name=wrong,
                    target_zone=wrong.replace("_cube", "_area"),
                    meta=meta,
                )
            if len(unplaced) > 1 and r < self.confusion_rate + self.forget_rate * 0.5:
                candidates = [n for n in unplaced if n != action.object_name]
                if candidates:
                    wrong = candidates[-1]
                    meta["noise"] = "confusion_wrong_pick"
                    return Action(
                        kind="grasp",
                        object_name=wrong,
                        target_zone=wrong.replace("_cube", "_area"),
                        meta=meta,
                    )

        if action.kind in ("place", "release"):
            if r < self.forget_rate:
                pool = [n for n in (placed + unplaced) if n]
                if not pool:
                    pool = [action.object_name]
                distractor = pool[self._step % len(pool)]
                meta["noise"] = "forget_regrasp"
                return Action(
                    kind="grasp",
                    object_name=distractor,
                    target_zone=distractor.replace("_cube", "_area"),
                    meta=meta,
                )
            if len(order) > 1 and r < self.forget_rate + self.confusion_rate * 0.5:
                # Place into a different color zone than intended.
                others = [n for n in order if n != action.object_name]
                wrong = others[self._step % len(others)]
                meta["noise"] = "confusion_wrong_place"
                return Action(
                    kind="place",
                    object_name=action.object_name,
                    target_zone=wrong.replace("_cube", "_area"),
                    meta=meta,
                )

        return action

    def _pseudo_rand(self) -> float:
        self._rng_seed = (1103515245 * (self._rng_seed + self._step + 1) + 12345) % (2**31)
        return (self._rng_seed % 10000) / 10000.0
