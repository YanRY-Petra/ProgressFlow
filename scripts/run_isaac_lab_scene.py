#!/usr/bin/env python3
"""
Assemble and run the ProgressFlow Isaac Lab scene.

Launch inside Isaac Lab::

    # From your Isaac Lab install:
    ./isaaclab.sh -p /path/to/ProgressFlow/scripts/run_isaac_lab_scene.py

    # Headless + dry scene check:
    ./isaaclab.sh -p /path/to/ProgressFlow/scripts/run_isaac_lab_scene.py --headless --mode scene

Modes
-----
  scene   — build the scene and step idle (assembly smoke test)
  demo    — progress-aware long-horizon pick-and-place
  baseline— myopic policy for contrast footage
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running without installing ProgressFlow as a package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProgressFlow Isaac Lab scene runner")
    parser.add_argument(
        "--mode",
        default="demo",
        choices=["scene", "demo", "baseline"],
        help="scene=assembly smoke test; demo/baseline=full policy loop",
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--control", default="ik_abs", choices=["ik_abs", "joint_pos"])
    parser.add_argument("--max_hl_steps", type=int, default=40, help="Max high-level policy steps")
    parser.add_argument("--idle_steps", type=int, default=300, help="Steps for --mode scene")
    parser.add_argument("--headless", action="store_true")
    # AppLauncher picks these up when Isaac Lab is present.
    return parser.parse_known_args()[0]


def _boot_isaac(args: argparse.Namespace):
    """Start Omniverse app via Isaac Lab AppLauncher when available."""
    try:
        from isaaclab.app import AppLauncher

        launcher_parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(launcher_parser)
        launch_args, _ = launcher_parser.parse_known_args()
        launch_args.headless = args.headless or getattr(launch_args, "headless", False)
        app_launcher = AppLauncher(launch_args)
        return app_launcher.app
    except Exception as exc:
        print(
            "[ProgressFlow] Could not start Isaac Lab AppLauncher.\n"
            "Run this script through Isaac Lab:\n"
            "  ./isaaclab.sh -p scripts/run_isaac_lab_scene.py\n"
            f"Details: {exc}"
        )
        sys.exit(1)


def run_scene_smoke(args: argparse.Namespace) -> None:
    """Instantiate scene cfg and idle-step (verifies asset assembly)."""
    import torch
    from isaaclab.envs import ManagerBasedRLEnv

    from progressflow.sim.isaac_lab.env_cfg import make_env_cfg
    from progressflow.sim.isaac_lab_env import describe_scene

    print(describe_scene())
    cfg = make_env_cfg(num_envs=args.num_envs, control_mode=args.control, device=args.device)
    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()
    print(f"[ok] Scene assembled. num_envs={env.num_envs} device={args.device}")

    # Zero / open-gripper hold to keep robot still.
    action_dim = env.action_space.shape[-1]
    zeros = torch.zeros((env.num_envs, action_dim), device=env.device)
    # For IK abs action: hold a safe pose above table; dim = 7 pose + 1 gripper
    if action_dim >= 8:
        # x,y,z above table center + down quat + open gripper
        zeros[:, 0] = 0.5
        zeros[:, 1] = 0.0
        zeros[:, 2] = 0.30
        zeros[:, 3] = 0.0
        zeros[:, 4] = 1.0
        zeros[:, 5] = 0.0
        zeros[:, 6] = 0.0
        zeros[:, 7] = 1.0

    for i in range(args.idle_steps):
        env.step(zeros)
        if i % 50 == 0:
            red = env.scene["red_cube"].data.root_pos_w[0].cpu().tolist()
            print(f"  step={i:04d}  red_cube_w={red}")

    env.close()
    print("[ok] Scene smoke test finished.")


def run_policy_demo(args: argparse.Namespace, use_progress: bool) -> None:
    from progressflow.policy import BaselinePolicy, DecisionNoiseWrapper, ProgressAwarePolicy
    from progressflow.sim.isaac_lab_env import IsaacLabConfig, IsaacLabEnvAdapter
    from progressflow.task_manager import TaskManager
    from progressflow.viz import render_demo_frame

    tm = TaskManager()
    print(tm.describe())
    print()

    adapter = IsaacLabEnvAdapter(
        IsaacLabConfig(
            control=args.control,
            num_envs=args.num_envs,
            device=args.device,
            headless=args.headless,
        ),
        task_manager=tm,
    )
    obs = adapter.reset()
    print(render_demo_frame(obs, "reset"))

    base = ProgressAwarePolicy() if use_progress else BaselinePolicy()
    policy = DecisionNoiseWrapper(
        base=base, confusion_rate=0.45, forget_rate=0.22, _rng_seed=7
    )
    policy.reset()

    done = False
    hl_steps = 0
    while not done and hl_steps < args.max_hl_steps:
        action = policy.act(obs, adapter.progress if use_progress else None)
        obs, done = adapter.step(action)
        hl_steps += 1
        print(render_demo_frame(obs, action.kind))
        print(f"[hl_step={hl_steps}] controller={obs.get('controller_phase')} done={done}")
        print()

    result = {
        "success": bool(adapter.progress and adapter.progress.is_done),
        "hl_steps": hl_steps,
        "sim_steps": adapter._step_count,
        "wrong_picks": adapter.wrong_picks,
        "repeated_picks": adapter.repeated_picks,
        "progress": adapter.progress.progress_value if adapter.progress else 0.0,
    }
    print("=== Isaac Lab Episode Result ===")
    print(json.dumps(result, indent=2))

    out = ROOT / "results" / "isaac_lab_timeline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(adapter.timeline, indent=2), encoding="utf-8")
    print(f"Saved timeline → {out}")
    adapter.close()


def main() -> None:
    args = parse_args()
    simulation_app = _boot_isaac(args)
    try:
        if args.mode == "scene":
            run_scene_smoke(args)
        elif args.mode == "demo":
            run_policy_demo(args, use_progress=True)
        else:
            run_policy_demo(args, use_progress=False)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
