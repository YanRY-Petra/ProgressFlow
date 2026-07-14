#!/usr/bin/env python3
"""
Validate Isaac Lab scene assembly files without requiring Isaac Sim.

Checks that the scene layout constants, configs, and scripts are present and
internally consistent with TableTopSim / Progress Manager naming.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REQUIRED = [
    "progressflow/sim/isaac_lab/scene_cfg.py",
    "progressflow/sim/isaac_lab/env_cfg.py",
    "progressflow/sim/isaac_lab/mdp.py",
    "progressflow/sim/isaac_lab/controller.py",
    "progressflow/sim/isaac_lab/__init__.py",
    "progressflow/sim/isaac_lab_env.py",
    "scripts/run_isaac_lab_scene.py",
    "configs/isaac_lab.yaml",
    "assets/scene_manifest.yaml",
]

EXPECTED_SYMBOLS = {
    "progressflow/sim/isaac_lab/scene_cfg.py": [
        "ProgressFlowSceneCfg",
        "make_cube_cfg",
        "make_zone_cfg",
        "finalize_scene_cfg",
        "CUBE_NAMES",
        "ZONE_NAMES",
    ],
    "progressflow/sim/isaac_lab/env_cfg.py": [
        "ProgressFlowEnvCfg",
        "ProgressFlowEnvCfg_PLAY",
        "make_env_cfg",
    ],
    "progressflow/sim/isaac_lab/controller.py": [
        "PickPlaceController",
        "pack_env_action",
    ],
    "progressflow/sim/isaac_lab_env.py": [
        "IsaacLabEnvAdapter",
        "IsaacLabConfig",
        "describe_scene",
    ],
}


def _parse_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def main() -> int:
    errors: list[str] = []
    for rel in REQUIRED:
        path = ROOT / rel
        if not path.exists():
            errors.append(f"missing file: {rel}")
            continue
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                errors.append(f"syntax error in {rel}: {exc}")
                continue
            for sym in EXPECTED_SYMBOLS.get(rel, []):
                names = _parse_names(path)
                if sym not in names:
                    errors.append(f"{rel}: expected symbol `{sym}`")
        elif path.suffix in {".yaml", ".yml"}:
            text = path.read_text(encoding="utf-8")
            if "red_cube" not in text and "Franka" not in text and "ik_abs" not in text:
                errors.append(f"{rel}: unexpected empty/unrelated config")

    # Naming consistency with TableTopSim / TaskManager
    from progressflow.task_manager import TaskManager

    tm = TaskManager()
    objects = [t.object_name for t in tm.parsed.subtasks]
    zones = [t.target_zone for t in tm.parsed.subtasks]
    if objects != ["red_cube", "blue_cube", "green_cube"]:
        errors.append(f"unexpected task objects: {objects}")
    if zones != ["red_area", "blue_area", "green_area"]:
        errors.append(f"unexpected task zones: {zones}")

    # Scene constants (string-level, no isaaclab import)
    scene_src = (ROOT / "progressflow/sim/isaac_lab/scene_cfg.py").read_text(encoding="utf-8")
    for token in [
        "red_cube",
        "blue_cube",
        "green_cube",
        "red_area",
        "blue_area",
        "green_area",
        "FRANKA_PANDA_HIGH_PD_CFG",
        "SeattleLabTable",
        "ProgressFlowSceneCfg",
    ]:
        if token not in scene_src:
            errors.append(f"scene_cfg.py missing token: {token}")

    if errors:
        print("Isaac Lab assembly validation FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("Isaac Lab assembly validation OK")
    print("  files   :", len(REQUIRED))
    print("  objects :", objects)
    print("  zones   :", zones)
    print("  launch  : ./isaaclab.sh -p scripts/run_isaac_lab_scene.py --mode scene")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
