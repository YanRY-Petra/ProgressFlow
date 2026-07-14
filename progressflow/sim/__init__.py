"""Simulation backends."""

from .tabletop import EpisodeResult, SimConfig, TableTopSim

__all__ = [
    "EpisodeResult",
    "SimConfig",
    "TableTopSim",
    "IsaacLabEnvAdapter",
    "IsaacLabConfig",
]


def __getattr__(name: str):
    if name in ("IsaacLabEnvAdapter", "IsaacLabConfig"):
        from .isaac_lab_env import IsaacLabConfig, IsaacLabEnvAdapter

        return IsaacLabEnvAdapter if name == "IsaacLabEnvAdapter" else IsaacLabConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
