from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch

DEFAULT_SLICE_RATIOS = (0.3, 0.7)

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)

def _to_2d(x):
    x = _to_numpy(x)
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3:
        x = x[0] if x.shape[0] == 1 else x[x.shape[0] // 2]
    return x

def minus1_1_to_uint8(x):
    x = _to_2d(x)
    x = np.clip((x + 1.0) / 2.0, 0.0, 1.0)
    return (x * 255.0).round().astype(np.uint8)

def save_png(x, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(minus1_1_to_uint8(x), mode="L").save(path)

def parse_case_slice(sample_name: str) -> Tuple[str, int]:
    sample_name = Path(str(sample_name)).stem
    if "_z" not in sample_name:
        raise ValueError(f"Invalid sample name without _z suffix: {sample_name}")
    case_id, slice_part = sample_name.rsplit("_z", 1)
    return case_id, int(slice_part)

def build_case_slice_targets(case_infos, ratios: Optional[Sequence[float]] = None):
    ratios = tuple(DEFAULT_SLICE_RATIOS if ratios is None else ratios)
    mapping = {}
    for case_id, info in case_infos.items():
        num_used = int(info["z_end"] - info["z_start"])
        if num_used <= 0:
            continue
        selected = []
        for ratio in ratios:
            ratio = min(max(float(ratio), 0.0), 1.0)
            pos = int(np.ceil(ratio * num_used)) - 1
            pos = min(max(pos, 0), num_used - 1)
            selected.append(int(info["z_start"] + pos))
        mapping[case_id] = sorted(set(selected))
    return mapping

class FixedSliceExporter:
    def __init__(self, save_root, case_infos, ratios: Optional[Sequence[float]] = None):
        self.save_root = Path(save_root)
        self.save_root.mkdir(parents=True, exist_ok=True)
        self.case_targets = build_case_slice_targets(case_infos, ratios)
        self.ratios = tuple(DEFAULT_SLICE_RATIOS if ratios is None else ratios)
        self.saved = set()

    def should_save(self, case_id: str, slice_idx: int) -> bool:
        return slice_idx in self.case_targets.get(case_id, []) and (case_id, slice_idx) not in self.saved

    def save(self, case_id: str, slice_idx: int, condition, target, prediction):
        if not self.should_save(case_id, slice_idx):
            return False
        sample_dir = self.save_root / case_id / f"z{int(slice_idx):03d}"
        save_png(condition, sample_dir / "condition.png")
        save_png(target, sample_dir / "target.png")
        save_png(prediction, sample_dir / "prediction.png")
        self.saved.add((case_id, int(slice_idx)))
        return True

    def summary(self):
        return {"cases": len(self.case_targets), "target_slices": sum(len(v) for v in self.case_targets.values()), "saved_slices": len(self.saved), "ratios": list(self.ratios)}
