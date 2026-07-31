#!/usr/bin/env python3
"""Run legacy YOLOFT video detection on the XS-VID test split."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

from checkpoint_utils import checkpoint_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)
    from ultralytics import YOLOFT
    from ultralytics.nn.autobackend import AutoBackend

    AutoBackend.warmup = lambda self, imgsz=(1, 3, 640, 640): None
    model = YOLOFT(str(args.config))
    model.model.load_state_dict(checkpoint_state(args.weights), strict=True)
    model.val(
        data=str(args.data),
        cfg=str(args.cfg),
        batch=1,
        device=[args.device],
        imgsz=1024,
        workers=4,
        half=True,
        split="test",
        save_json=True,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )
    candidates = glob.glob(str(args.project / args.name / "*.json"))
    if not candidates:
        raise FileNotFoundError(f"No JSON prediction file under {args.project / args.name}")
    print(f"XSVID_VID_EVAL_OK {max(candidates, key=os.path.getmtime)}", flush=True)


if __name__ == "__main__":
    main()
