"""Isaac Lab specific scene / env / controller package.

Import only when Isaac Lab + Isaac Sim are available::

    from progressflow.sim.isaac_lab.env_cfg import make_env_cfg
    from progressflow.sim.isaac_lab.scene_cfg import ProgressFlowSceneCfg
"""

from __future__ import annotations

__all__ = [
    "ProgressFlowSceneCfg",
    "ProgressFlowEnvCfg",
    "make_env_cfg",
    "PickPlaceController",
    "finalize_scene_cfg",
]


def __getattr__(name: str):
    # Lazy exports so `import progressflow.sim.isaac_lab` can be probed safely.
    if name == "ProgressFlowSceneCfg":
        from .scene_cfg import ProgressFlowSceneCfg as cls

        return cls
    if name == "ProgressFlowEnvCfg":
        from .env_cfg import ProgressFlowEnvCfg as cls

        return cls
    if name == "make_env_cfg":
        from .env_cfg import make_env_cfg as fn

        return fn
    if name == "PickPlaceController":
        from .controller import PickPlaceController as cls

        return cls
    if name == "finalize_scene_cfg":
        from .scene_cfg import finalize_scene_cfg as fn

        return fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
