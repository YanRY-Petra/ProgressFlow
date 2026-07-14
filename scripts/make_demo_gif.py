#!/usr/bin/env python3
"""
Render a short demo animation (GIF) of the progress-aware long-horizon run.

Produces demo.gif at the repository root for README / 1-minute video storyboard.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from progressflow.policy import ProgressAwarePolicy
from progressflow.sim import SimConfig, TableTopSim
from progressflow.task_manager import TaskManager


def _frame_image(text: str, size=(960, 540)):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", size, (18, 22, 28))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 18)
        title = ImageFont.truetype("DejaVuSans.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
        title = font

    draw.rectangle([0, 0, size[0], 56], fill=(28, 80, 140))
    draw.text((24, 14), "ProgressFlow  ·  Long-Horizon Sequential Manipulation", fill=(240, 244, 248), font=title)

    y = 80
    for line in text.splitlines():
        draw.text((28, y), line, fill=(220, 228, 235), font=font)
        y += 22
        if y > size[1] - 40:
            break
    return img


def main() -> None:
    from progressflow.viz import render_demo_frame

    tm = TaskManager()
    sim = TableTopSim(tm, SimConfig(max_steps=40, seed=7))
    policy = ProgressAwarePolicy()
    obs = sim.reset()

    frames = [_frame_image(render_demo_frame(obs, "reset"))]
    done = False
    while not done:
        action = policy.act(obs, sim.progress)
        obs, done = sim.step(action)
        frames.append(_frame_image(render_demo_frame(obs, action.kind)))
        # Hold key milestones a bit longer
        if obs["progress"]["progress_percent"] in (33, 50, 66, 100) or action.kind == "place":
            frames.append(frames[-1])
            frames.append(frames[-1])

    out = ROOT / "demo.gif"
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=350,
        loop=0,
    )
    print(f"Wrote {out} ({len(frames)} frames)")
    # Also keep a timeline artifact
    timeline = ROOT / "results" / "demo_timeline.json"
    import json

    timeline.parent.mkdir(parents=True, exist_ok=True)
    timeline.write_text(json.dumps(sim.timeline, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
