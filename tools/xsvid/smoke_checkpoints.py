#!/usr/bin/env python3
"""Load the three released legacy checkpoints against this repository only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from checkpoint_utils import checkpoint_state


def report_load(name: str, module: torch.nn.Module, weights: Path) -> dict:
    missing, unexpected = module.load_state_dict(checkpoint_state(weights), strict=True)
    result = {"checkpoint": str(weights), "missing": len(missing), "unexpected": len(unexpected)}
    print(f"{name}: missing={result['missing']} unexpected={result['unexpected']}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    temporal_config = args.model_package / "configs" / "yoloft-l-temporal.yaml"
    checkpoints = args.model_package / "checkpoints"

    from ultralytics import YOLOFT
    from ultralytics.nn.omni import VideoOmniModel
    from ultralytics.nn.sot_transt import YOLOFTSotTransT

    result = {}
    detector = YOLOFT(str(temporal_config)).model
    result["vid"] = report_load("VID", detector, checkpoints / "yoloft-l-xsvid-v2-paper.pt")
    mot = VideoOmniModel(str(temporal_config), ch=3, nc=7, verbose=False, embed_dim=256,
                         emb_stride=4, use_pre_mstf=False, emb_src_hi=20, feat_mode="simple")
    result["mot"] = report_load("MOT", mot, checkpoints / "yoloft-l-xsvid-v2-mot.pt")
    sot = YOLOFTSotTransT(str(temporal_config), 3, 7, verbose=False, feat_mode="fuse918", embed=256, n_fusion=4)
    result["sot"] = report_load("SOT", sot, checkpoints / "yoloft-l-xsvid-v2-sot.pt")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"SMOKE_CHECKPOINTS_OK {args.output}", flush=True)


if __name__ == "__main__":
    main()
