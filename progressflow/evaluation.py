"""
Evaluation: Baseline (prompt-only) vs Progress-aware,
under the *same* decision-side noise.

Metrics
-------
- Task Success Rate
- Average Completion
- Wrong Pick
- Repeated Pick
- Completion Time (steps)
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .policy import BaselinePolicy, DecisionNoiseWrapper, ProgressAwarePolicy
from .sim import SimConfig, TableTopSim
from .task_manager import DEFAULT_INSTRUCTION, TaskManager

# Shared decision noise (identical for both methods).
DEFAULT_CONFUSION = 0.45
DEFAULT_FORGET = 0.22


@dataclass
class AggregateMetrics:
    method: str
    episodes: int
    task_success_rate: float
    average_completion: float
    avg_wrong_picks: float
    avg_repeated_picks: float
    avg_completion_time: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_evaluation(
    n_episodes: int = 20,
    max_steps: int = 80,
    instruction: str = DEFAULT_INSTRUCTION,
    seed0: int = 0,
    confusion_rate: float = DEFAULT_CONFUSION,
    forget_rate: float = DEFAULT_FORGET,
) -> dict[str, AggregateMetrics]:
    """
    Fair ablation:
      - same decision noise (confusion/forget) on both methods
      - only difference: ProgressManager vs prompt-only baseline
    """
    task_manager = TaskManager(instruction)
    method_names = ("baseline", "progress_aware")
    aggregates: dict[str, AggregateMetrics] = {}

    for name in method_names:
        successes = 0
        completions = 0.0
        wrong = 0.0
        repeated = 0.0
        times = 0.0
        use_progress = name == "progress_aware"

        for ep in range(n_episodes):
            sim = TableTopSim(
                task_manager,
                SimConfig(max_steps=max_steps, seed=seed0 + ep * 17),
            )
            # Same noise seed schedule for both methods at episode ep.
            base = BaselinePolicy() if name == "baseline" else ProgressAwarePolicy()
            policy = DecisionNoiseWrapper(
                base=base,
                confusion_rate=confusion_rate,
                forget_rate=forget_rate,
            )
            policy._rng_seed = seed0 + ep * 31
            result = sim.run_episode(policy, use_progress=use_progress)
            successes += int(result.success)
            completions += result.completed_subtasks / max(result.total_subtasks, 1)
            wrong += result.wrong_picks
            repeated += result.repeated_picks
            times += result.steps

        aggregates[name] = AggregateMetrics(
            method=name,
            episodes=n_episodes,
            task_success_rate=successes / n_episodes,
            average_completion=completions / n_episodes,
            avg_wrong_picks=wrong / n_episodes,
            avg_repeated_picks=repeated / n_episodes,
            avg_completion_time=times / n_episodes,
        )
    return aggregates


def metrics_to_markdown(metrics: dict[str, AggregateMetrics]) -> str:
    headers = [
        "Method",
        "Success Rate",
        "Avg Completion",
        "Wrong Pick",
        "Repeated Pick",
        "Avg Steps",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for m in metrics.values():
        lines.append(
            "| "
            + " | ".join(
                [
                    m.method,
                    f"{m.task_success_rate:.2f}",
                    f"{m.average_completion:.2f}",
                    f"{m.avg_wrong_picks:.2f}",
                    f"{m.avg_repeated_picks:.2f}",
                    f"{m.avg_completion_time:.1f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def save_results(
    metrics: dict[str, AggregateMetrics],
    out_dir: str | Path,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = {k: v.to_dict() for k, v in metrics.items()}
    json_path = out / "evaluation.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = out / "evaluation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(payload.values())).keys()))
        writer.writeheader()
        for row in payload.values():
            writer.writerow(row)

    md_path = out / "evaluation.md"
    md_path.write_text(
        "# ProgressFlow Evaluation\n\n"
        "Same decision noise on both methods "
        f"(confusion={DEFAULT_CONFUSION}, forget={DEFAULT_FORGET}).\n\n"
        + metrics_to_markdown(metrics)
        + "\n",
        encoding="utf-8",
    )
    return json_path


def main() -> None:
    metrics = run_evaluation(n_episodes=30)
    print(metrics_to_markdown(metrics))
    path = save_results(metrics, Path(__file__).resolve().parents[1] / "results")
    print(f"\nSaved results to {path}")


if __name__ == "__main__":
    main()
