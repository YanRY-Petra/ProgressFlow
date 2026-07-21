"""
ProgressFlow entry point.

Demo pipeline
-------------
Isaac Lab (or TableTopSim)
  → Franka Panda
  → Table + 3 Cubes + 3 Target Zones
  → Language Instruction
  → Robot Policy
  → Progress State
  → Visualization
  → Evaluation
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from progressflow.evaluation import metrics_to_markdown, run_evaluation, save_results
from progressflow.policy import BaselinePolicy, DecisionNoiseWrapper, ProgressAwarePolicy
from progressflow.sim import SimConfig, TableTopSim
from progressflow.task_manager import DEFAULT_INSTRUCTION, TaskManager
from progressflow.viz import render_demo_frame


ROOT = Path(__file__).resolve().parent

# Demo uses the same shared decision noise as evaluation.
DEMO_CONFUSION = 0.45
DEMO_FORGET = 0.22


def _demo_policy(policy_name: str) -> tuple[Any, bool]:
    base = BaselinePolicy() if policy_name == "baseline" else ProgressAwarePolicy()
    policy = DecisionNoiseWrapper(
        base=base,
        confusion_rate=DEMO_CONFUSION,
        forget_rate=DEMO_FORGET,
        _rng_seed=7,
    )
    return policy, policy_name != "baseline"

def run_demo(
    backend: str = "sim",
    policy_name: str = "progress_aware",
    sleep: float = 0.35,
    max_steps: int = 40,
    save_timeline: bool = True,
) -> None:
    if backend == "isaac_lab":
        from progressflow.sim.isaac_lab_env import IsaacLabConfig, IsaacLabEnvAdapter

        adapter = IsaacLabEnvAdapter(IsaacLabConfig())
        if not adapter.available():
            print(
                "[ProgressFlow] Isaac Lab not available — falling back to TableTopSim.\n"
                "On a machine with Isaac Lab installed, run:\n"
                "  ./isaaclab.sh -p scripts/run_isaac_lab_scene.py --mode demo\n"
            )
        else:
            print(
                "[ProgressFlow] Prefer launching via Isaac Lab AppLauncher:\n"
                "  ./isaaclab.sh -p scripts/run_isaac_lab_scene.py --mode demo\n"
                "Attempting in-process adapter anyway...\n"
            )
            _run_isaac_adapter_demo(adapter, policy_name, max_steps, save_timeline)
            return

    task_manager = TaskManager(DEFAULT_INSTRUCTION)
    print(task_manager.describe())
    print()

    sim = TableTopSim(task_manager, SimConfig(max_steps=max_steps, seed=7))
    policy, use_progress = _demo_policy(policy_name)
    obs = sim.reset()
    print(render_demo_frame(obs, action_kind="reset"))
    print("\n--- starting long-horizon execution ---\n")

    done = False
    frames: list[str] = []
    while not done:
        action = policy.act(obs, sim.progress if use_progress else None)
        obs, done = sim.step(action)
        frame = render_demo_frame(obs, action_kind=action.kind)
        print(frame)
        print()
        frames.append(frame)
        if sleep > 0:
            time.sleep(sleep)

    assert sim.progress is not None
    result = {
        "success": sim.progress.is_done,
        "steps": sim.step_count,
        "progress": sim.progress.progress_value,
        "wrong_picks": sim.wrong_picks,
        "repeated_picks": sim.repeated_picks,
    }
    print("=== Episode Result ===")
    print(json.dumps(result, indent=2))

    if save_timeline:
        out = ROOT / "results" / "demo_timeline.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(sim.timeline, indent=2), encoding="utf-8")
        log_path = ROOT / "results" / "demo_frames.txt"
        log_path.write_text("\n\n==== FRAME ====\n\n".join(frames), encoding="utf-8")
        print(f"Saved timeline → {out}")
        print(f"Saved frames   → {log_path}")


def _run_isaac_adapter_demo(
    adapter,
    policy_name: str,
    max_steps: int,
    save_timeline: bool,
) -> None:
    from progressflow.viz import render_demo_frame as _render

    obs = adapter.reset()
    print(_render(obs, "reset"))
    policy, use_progress = _demo_policy(policy_name)
    policy.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        action = policy.act(obs, adapter.progress if use_progress else None)
        obs, done = adapter.step(action)
        steps += 1
        print(_render(obs, action.kind))
        print()
    result = {
        "success": bool(adapter.progress and adapter.progress.is_done),
        "steps": adapter._step_count,
        "progress": adapter.progress.progress_value if adapter.progress else 0.0,
        "wrong_picks": adapter.wrong_picks,
        "repeated_picks": adapter.repeated_picks,
    }
    print("=== Isaac Lab Episode Result ===")
    print(json.dumps(result, indent=2))
    if save_timeline:
        out = ROOT / "results" / "isaac_lab_timeline.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(adapter.timeline, indent=2), encoding="utf-8")
        print(f"Saved timeline → {out}")
    adapter.close()


def run_eval_cli(n_episodes: int) -> None:
    metrics = run_evaluation(n_episodes=n_episodes)
    print(metrics_to_markdown(metrics))
    path = save_results(metrics, ROOT / "results")
    print(f"\nSaved results to {path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ProgressFlow — progress-aware long-horizon demo")
    p.add_argument(
        "mode",
        nargs="?",
        default="demo",
        choices=["demo", "eval", "both"],
        help="Run interactive demo, evaluation, or both",
    )
    p.add_argument("--backend", default="sim", choices=["sim", "isaac_lab"])
    p.add_argument("--policy", default="progress_aware", choices=["progress_aware", "baseline"])
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--sleep", type=float, default=0.25, help="Demo frame delay (seconds)")
    p.add_argument("--max-steps", type=int, default=40)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.mode in ("demo", "both"):
        run_demo(
            backend=args.backend,
            policy_name=args.policy,
            sleep=args.sleep,
            max_steps=args.max_steps,
        )
    if args.mode in ("eval", "both"):
        if args.mode == "both":
            print("\n" + "=" * 60 + "\nEvaluation\n" + "=" * 60 + "\n")
        run_eval_cli(args.episodes)


if __name__ == "__main__":
    main()
