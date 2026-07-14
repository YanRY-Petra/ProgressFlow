# ProgressFlow — Technical Summary

## Problem

Sequential tabletop instructions (e.g. “move red, then blue, then green”) are **long-horizon**. A flat policy that maps `observation → action` without task memory fails with wrong picks, repeated picks, and stalled progress.

## Claim

Exposing an explicit **progress state** (current / completed / remaining subtasks + scalar progress value) measurably improves sequential manipulation reliability — even when the underlying low-level skill is a simple rule-based pick-and-place controller.

## Method

1. **Task Parser** converts the language instruction into ordered `Subtask`s.
2. **Progress Manager** tracks status `{pending, active, grasped, completed}` and `progress_value ∈ [0,1]`.
3. **Progress-aware Policy** conditions each `Observe → Pick → Transport → Place` cycle on the current subtask.
4. **Baseline Policy** uses the same observation space but **ignores** progress — modeling memoryless failure modes.
5. **Evaluation** reports Success Rate, Avg Completion, Wrong/Repeated Picks, and Steps.

## System

- **Preferred platform:** Isaac Lab + Franka Panda (modern robot-learning workflow).
- **Isaac Lab assembly:** `progressflow/sim/isaac_lab/scene_cfg.py` builds
  Franka + SeattleLabTable + 3 colored blocks + 3 visual target pads; absolute IK
  control via `ProgressFlowEnvCfg`; `PickPlaceController` maps ProgressFlow
  Actions to EE pose / gripper commands. Launch with
  `scripts/run_isaac_lab_scene.py`.
- **Default runnable backend:** `TableTopSim` (pure Python) with the same APIs for laptop / CI demos.
- **Visualization:** HUD with instruction, progress bar, and current target (PALM-like progress value).

## Results (run locally)

```bash
python main.py eval --episodes 30
```

Progress-aware control is expected to approach **~1.0 success** with **near-zero wrong/repeated picks**, while the baseline degrades as horizon / confusion increases.

## Limitations

- Progress is rule-based, not learned from video/language.
- TableTopSim abstracts contact-rich control; Isaac Lab path uses scripted abs-IK primitives (not a trained RL policy).
- Running the Isaac Lab scene requires a local Isaac Lab / Isaac Sim install and Nucleus assets.

## Next Steps

Replace `ProgressManager` updates with a learned progress predictor; keep the same interfaces for policy, HUD, and evaluation.
