#!/usr/bin/env python3
"""Run the native legacy VID checkpoint on real XS-VID v2 test frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch

from checkpoint_utils import checkpoint_state


def tensor_from_image(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).cuda()


def describe(value):
    if isinstance(value, torch.Tensor):
        return {"type": "tensor", "shape": list(value.shape)}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "items": [describe(item) for item in value]}
    if isinstance(value, dict):
        return {"type": "dict", "keys": sorted(value)}
    return {"type": type(value).__name__}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from ultralytics import YOLOFT

    annotation = json.loads((args.data_root / "annotations" / "test.json").read_text(encoding="utf-8"))
    first_video = min(image["video_id"] for image in annotation["images"])
    frames = sorted(
        (image for image in annotation["images"] if image["video_id"] == first_video),
        key=lambda image: image["frame_index"],
    )[: args.frames]
    model = YOLOFT(str(args.config)).model.cuda().eval()
    model.load_state_dict(checkpoint_state(args.checkpoint), strict=True)
    outputs = []
    states = [None, None, None]
    with torch.no_grad():
        for frame in frames:
            prediction, states = model.predict(
                tensor_from_image(args.data_root / "images" / frame["file_name"]), *states
            )
            outputs.append(
                {
                    "image_id": frame["id"],
                    "prediction": describe(prediction),
                    "states": describe(states),
                }
            )
    record = {"model_class": type(model).__name__, "video_id": first_video, "frames": outputs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"SMOKE_VID_FORWARD_OK {args.output}")


if __name__ == "__main__":
    main()
