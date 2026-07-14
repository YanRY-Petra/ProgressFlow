"""Thin re-export / CLI for evaluation."""

from progressflow.evaluation import (
    AggregateMetrics,
    main,
    metrics_to_markdown,
    run_evaluation,
    save_results,
)

__all__ = [
    "AggregateMetrics",
    "metrics_to_markdown",
    "run_evaluation",
    "save_results",
]


if __name__ == "__main__":
    main()
