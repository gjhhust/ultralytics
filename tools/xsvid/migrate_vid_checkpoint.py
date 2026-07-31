#!/usr/bin/env python3
"""Convert the historical YOLOFT VID checkpoint to the release runtime graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from checkpoint_utils import checkpoint_state, migrate_index_shift


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from ultralytics import YOLOFT

    detector = YOLOFT(str(args.config)).model
    migrated, transfer = migrate_index_shift(detector, checkpoint_state(args.source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": migrated,
            "meta": {
                "format": "legacy_yoloft_temporal_state_dict",
                "source_sha256": sha256(args.source),
                "migration": transfer,
            },
        },
        args.output,
    )
    report = {
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "migration": transfer,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"MIGRATE_VID_CHECKPOINT_OK {args.report}", flush=True)


if __name__ == "__main__":
    main()
