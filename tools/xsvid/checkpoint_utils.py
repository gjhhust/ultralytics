"""Checkpoint helpers shared by the XS-VID legacy release tools."""

from __future__ import annotations

from pathlib import Path

import torch


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    """Return a plain state dictionary from supported legacy payloads."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        value = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    else:
        value = checkpoint
    if hasattr(value, "state_dict"):
        value = value.state_dict()
    if not isinstance(value, dict) or not all(isinstance(item, torch.Tensor) for item in value.values()):
        raise TypeError(f"Unsupported checkpoint payload in {path}: {type(value)!r}")
    return value


def migrate_index_shift(
    module: torch.nn.Module, source: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, int | str]]:
    """Map the historical unshifted graph onto the equivalent temporal graph."""
    target = module.state_dict()

    def suffix(key: str) -> str:
        parts = key.split(".")
        if len(parts) > 2 and parts[0] == "model" and parts[1].isdigit():
            return ".".join(parts[2:])
        return key

    source_items = list(source.items())
    target_items = list(target.items())
    source_index = 0
    target_index = 0
    migrated: dict[str, torch.Tensor] = {}
    while source_index < len(source_items) and target_index < len(target_items):
        source_key, source_value = source_items[source_index]
        target_key, target_value = target_items[target_index]
        if suffix(source_key) == suffix(target_key) and source_value.shape == target_value.shape:
            migrated[target_key] = source_value
            source_index += 1
            target_index += 1
        elif suffix(source_key) == suffix(target_key):
            source_index += 1
            target_index += 1
        else:
            target_index += 1

    if len(migrated) != len(target):
        raise RuntimeError(f"Historical state migration matched {len(migrated)}/{len(target)} tensors")
    module.load_state_dict(migrated, strict=True)
    return migrated, {"mode": "index_shift", "matched": len(migrated), "total": len(target)}
