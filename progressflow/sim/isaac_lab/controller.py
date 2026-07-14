"""
Scripted pick-and-place controller bridging ProgressFlow Actions → Isaac Lab.

Converts high-level ``Action(kind, object_name, target_zone)`` into absolute
EE pose + gripper binary commands for Differential IK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .scene_cfg import CUBE_NAMES, ZONE_INIT_POS


# Downward gripper orientation (w, x, y, z) — fingers pointing -Z.
DOWN_ORIENTATION = (0.0, 1.0, 0.0, 0.0)

PRE_GRASP_HEIGHT = 0.15
GRASP_HEIGHT = 0.03
PLACE_HEIGHT = 0.04
TRANSPORT_HEIGHT = 0.18


@dataclass
class ControllerState:
    phase: str = "idle"
    object_name: str | None = None
    target_zone: str | None = None
    hold_steps: int = 0
    last_command: dict[str, Any] = field(default_factory=dict)


class PickPlaceController:
    """
    Finite-state motion primitives:

      approach → lower → close → lift → transport → lower → open → retreat
    """

    def __init__(self, device: str = "cuda:0", settle_steps: int = 8):
        self.device = device
        self.settle_steps = settle_steps
        self.state = ControllerState()

    def reset(self) -> None:
        self.state = ControllerState()

    def set_goal(self, kind: str, object_name: str | None, target_zone: str | None) -> None:
        """Start a new primitive from a ProgressFlow policy action."""
        if kind in ("idle", "observe"):
            self.state.phase = "idle"
            return
        self.state.object_name = object_name
        self.state.target_zone = target_zone
        self.state.hold_steps = 0
        if kind in ("approach", "grasp", "pick"):
            self.state.phase = "approach"
        elif kind == "transport":
            self.state.phase = "transport"
        elif kind in ("place", "release"):
            self.state.phase = "place_lower"
        else:
            self.state.phase = "approach"

    def compute(
        self,
        cube_pos_w: dict[str, torch.Tensor],
        env_origins: torch.Tensor,
        ee_pos_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Returns
        -------
        arm_action : (N, 7)  absolute pose [x,y,z, qw,qx,qy,qz]
        gripper    : (N, 1)  +1 open / -1 close  (BinaryJointPositionAction convention)
        done_primitive : bool — current high-level action finished
        """
        n = env_origins.shape[0]
        arm = torch.zeros((n, 7), device=self.device)
        # default: hover over table center, gripper open
        default_pos = env_origins + torch.tensor((0.5, 0.0, 0.25), device=self.device)
        arm[:, :3] = default_pos
        arm[:, 3] = DOWN_ORIENTATION[0]
        arm[:, 4] = DOWN_ORIENTATION[1]
        arm[:, 5] = DOWN_ORIENTATION[2]
        arm[:, 6] = DOWN_ORIENTATION[3]
        gripper = torch.ones((n, 1), device=self.device)  # open
        done = False

        obj = self.state.object_name
        zone = self.state.target_zone
        if self.state.phase == "idle" or obj is None:
            return arm, gripper, True

        obj_pos = cube_pos_w.get(obj)
        if obj_pos is None:
            return arm, gripper, True

        color = (zone or obj.replace("_cube", "_area")).replace("_area", "")
        zone_local = ZONE_INIT_POS.get(color, (0.5, 0.25, 0.002))
        zone_pos = env_origins + torch.tensor(zone_local, device=self.device)

        phase = self.state.phase
        if phase == "approach":
            target = obj_pos.clone()
            target[:, 2] = env_origins[:, 2] + PRE_GRASP_HEIGHT
            arm[:, :3] = target
            gripper[:] = 1.0
            if self._reached(ee_pos_w, target):
                self.state.phase = "grasp_lower"
                self.state.hold_steps = 0

        elif phase == "grasp_lower":
            target = obj_pos.clone()
            target[:, 2] = env_origins[:, 2] + GRASP_HEIGHT
            arm[:, :3] = target
            gripper[:] = 1.0
            if self._reached(ee_pos_w, target):
                self.state.phase = "grasp_close"
                self.state.hold_steps = 0

        elif phase == "grasp_close":
            target = obj_pos.clone()
            target[:, 2] = env_origins[:, 2] + GRASP_HEIGHT
            arm[:, :3] = target
            gripper[:] = -1.0  # close
            self.state.hold_steps += 1
            if self.state.hold_steps >= self.settle_steps:
                self.state.phase = "lift"
                self.state.hold_steps = 0
                done = True  # high-level "grasp" complete

        elif phase == "lift":
            target = obj_pos.clone()
            target[:, 2] = env_origins[:, 2] + TRANSPORT_HEIGHT
            arm[:, :3] = target
            gripper[:] = -1.0
            if self._reached(ee_pos_w, target):
                self.state.phase = "transport"
                self.state.hold_steps = 0

        elif phase == "transport":
            target = zone_pos.clone()
            target[:, 2] = env_origins[:, 2] + TRANSPORT_HEIGHT
            arm[:, :3] = target
            gripper[:] = -1.0
            if self._reached(ee_pos_w, target):
                self.state.phase = "place_lower"
                self.state.hold_steps = 0
                done = True  # high-level "transport" complete

        elif phase == "place_lower":
            target = zone_pos.clone()
            target[:, 2] = env_origins[:, 2] + PLACE_HEIGHT
            arm[:, :3] = target
            gripper[:] = -1.0
            if self._reached(ee_pos_w, target):
                self.state.phase = "place_open"
                self.state.hold_steps = 0

        elif phase == "place_open":
            target = zone_pos.clone()
            target[:, 2] = env_origins[:, 2] + PLACE_HEIGHT
            arm[:, :3] = target
            gripper[:] = 1.0  # open
            self.state.hold_steps += 1
            if self.state.hold_steps >= self.settle_steps:
                self.state.phase = "retreat"
                self.state.hold_steps = 0

        elif phase == "retreat":
            target = zone_pos.clone()
            target[:, 2] = env_origins[:, 2] + PRE_GRASP_HEIGHT
            arm[:, :3] = target
            gripper[:] = 1.0
            if self._reached(ee_pos_w, target):
                self.state.phase = "idle"
                done = True  # high-level "place" complete

        self.state.last_command = {"phase": self.state.phase, "object": obj, "zone": zone}
        return arm, gripper, done

    def _reached(self, ee: torch.Tensor, target: torch.Tensor, thresh: float = 0.025) -> bool:
        dist = torch.linalg.norm(ee - target, dim=-1)
        return bool((dist < thresh).all().item())


def pack_env_action(arm: torch.Tensor, gripper: torch.Tensor) -> torch.Tensor:
    """Concatenate arm (N,7) + gripper (N,1) → (N, 8) for ManagerBasedRLEnv."""
    return torch.cat([arm, gripper], dim=-1)


def cube_positions_from_scene(scene, device: str) -> dict[str, torch.Tensor]:
    out = {}
    for name in CUBE_NAMES:
        out[name] = scene[name].data.root_pos_w[:, :3].to(device)
    return out
