from pathlib import Path
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import SimpleITK as sitk


class SynthRAD2p5DDataset(Dataset):
    def __init__(self, root, task="task1", anatomy="HN", split="train", radius=1, target_size=(192, 256), cache_volumes=True, ct_clip=(-1000.0, 2000.0), return_meta=False, keep_ratio=1.0, mask_margin=10):
        super().__init__()
        print(f"\n[{task} - {anatomy} - {split}] 处理数据中...")
        assert task in ["task1", "task2"]
        assert anatomy in ["AB", "HN", "TH"]
        assert split in ["train", "val", "test"]
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio 必须在 (0, 1] 范围内，当前为: {keep_ratio}")
        self.root = Path(root)
        self.task = task
        self.anatomy = anatomy
        self.split = split
        self.radius = radius
        self.target_size = target_size
        self.cache_volumes = cache_volumes
        self.ct_clip = ct_clip
        self.return_meta = return_meta
        self.keep_ratio = float(keep_ratio)
        self.mask_margin = mask_margin
        self.split_dir = self.root / self.task / self.anatomy / self.split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"split_dir 不存在: {self.split_dir}")
        self.input_filename = "mr.mha" if self.task == "task1" else "cbct.mha"
        self.target_filename = "ct.mha"
        self.mask_filename = "mask.mha"
        self.case_dirs = sorted([p for p in self.split_dir.iterdir() if p.is_dir()])
        if len(self.case_dirs) == 0:
            raise RuntimeError(f"{self.split_dir} 下没有找到病例目录")
        self.case_infos = {}
        self.samples = []
        self.cache = {}
        self._build_index()

    def _read_mha(self, path):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)

    def _normalize_input(self, vol):
        mask = vol > 0
        out = np.full_like(vol, fill_value=-1.0, dtype=np.float32)
        if mask.sum() == 0:
            return out
        fg = vol[mask]
        lo = np.percentile(fg, 1)
        hi = np.percentile(fg, 99)
        if hi - lo < 1e-6:
            out[mask] = 0.0
            return out
        fg = np.clip(fg, lo, hi)
        out[mask] = (2.0 * (fg - lo) / (hi - lo) - 1.0).astype(np.float32)
        return out

    def _normalize_ct(self, vol):
        lo, hi = self.ct_clip
        vol = np.clip(vol.astype(np.float32), lo, hi)
        return (2.0 * (vol - lo) / (hi - lo) - 1.0).astype(np.float32)

    def _normalize_cbct(self, vol, mask=None):
        lo, hi = self.ct_clip
        if mask is None:
            vol = np.clip(vol.astype(np.float32), lo, hi)
            return (2.0 * (vol - lo) / (hi - lo) - 1.0).astype(np.float32)
        vol = vol.astype(np.float32)
        out = np.full_like(vol, fill_value=-1.0, dtype=np.float32)
        mask = mask > 0
        if mask.sum() == 0:
            return out
        fg = np.clip(vol[mask], lo, hi)
        out[mask] = (2.0 * (fg - lo) / (hi - lo) - 1.0).astype(np.float32)
        return out

    def _get_3d_bounding_box(self, mask_vol, H, W):
        mask_2d = np.any(mask_vol > 0, axis=0)
        if not np.any(mask_2d):
            return 0, H, 0, W
        rows = np.any(mask_2d, axis=1)
        cols = np.any(mask_2d, axis=0)
        h_min, h_max = np.where(rows)[0][[0, -1]]
        w_min, w_max = np.where(cols)[0][[0, -1]]
        return max(0, h_min - self.mask_margin), min(H, h_max + self.mask_margin + 1), max(0, w_min - self.mask_margin), min(W, w_max + self.mask_margin + 1)

    def _resize_slice(self, x, is_mask=False):
        if self.target_size is None:
            return x
        x = x.unsqueeze(0)
        mode = "nearest" if is_mask else "bilinear"
        kwargs = {} if is_mask else {"align_corners": False, "antialias": True}
        return F.interpolate(x, size=self.target_size, mode=mode, **kwargs).squeeze(0)

    def _load_case(self, case_dir):
        x = self._read_mha(case_dir / self.input_filename)
        y = self._read_mha(case_dir / self.target_filename)
        mask = self._read_mha(case_dir / self.mask_filename)
        if x.shape != y.shape or x.shape != mask.shape:
            raise ValueError(f"尺寸不一致: x={x.shape}, y={y.shape}, mask={mask.shape}")
        _, H, W = x.shape
        h_min, h_max, w_min, w_max = self._get_3d_bounding_box(mask, H, W)
        x_crop = x[:, h_min:h_max, w_min:w_max]
        y_crop = y[:, h_min:h_max, w_min:w_max]
        mask_crop = mask[:, h_min:h_max, w_min:w_max]
        if self.task == "task2":
            x_norm = self._normalize_cbct(x_crop, mask_crop)
        else:
            x_norm = self._normalize_input(x_crop)
        return x_norm, self._normalize_ct(y_crop), mask_crop

    def _get_center_slice_range(self, depth):
        valid_start = self.radius
        valid_end = depth - self.radius
        if valid_end <= valid_start:
            return None, None
        if self.keep_ratio >= 1.0:
            return valid_start, valid_end
        num_keep = max(1, int(round(depth * self.keep_ratio)))
        center_start = (depth - num_keep) // 2
        center_end = center_start + num_keep
        z_start = max(valid_start, center_start)
        z_end = min(valid_end, center_end)
        if z_end <= z_start:
            return None, None
        return z_start, z_end

    def _build_index(self):
        for case_dir in self.case_dirs:
            case_id = case_dir.name
            try:
                x, y, mask = self._load_case(case_dir)
            except Exception as e:
                warnings.warn(f"跳过病例 {case_id}: {e}")
                continue
            depth = x.shape[0]
            if depth < 2 * self.radius + 1:
                continue
            z_start, z_end = self._get_center_slice_range(depth)
            if z_start is None or z_end is None:
                continue
            self.case_infos[case_id] = {"case_dir": case_dir, "depth": depth, "z_start": z_start, "z_end": z_end, "num_used_slices": z_end - z_start}
            if self.cache_volumes:
                self.cache[case_id] = {"x": x, "y": y, "mask": mask}
            for z in range(z_start, z_end):
                self.samples.append((case_id, z))
        if len(self.samples) == 0:
            raise RuntimeError("没有构造出任何有效样本！")
        print(f"[SynthRAD2p5DDataset] 准备就绪: {len(self.case_infos)} 个有效病例, {len(self.samples)} 个总样本切片")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        case_id, z = self.samples[idx]
        if self.cache_volumes:
            x = self.cache[case_id]["x"]
            y = self.cache[case_id]["y"]
            mask = self.cache[case_id]["mask"]
        else:
            x, y, mask = self._load_case(self.case_infos[case_id]["case_dir"])
        x_prev = self._resize_slice(torch.from_numpy(x[z - 1]).unsqueeze(0).float())
        x_center = self._resize_slice(torch.from_numpy(x[z]).unsqueeze(0).float())
        x_next = self._resize_slice(torch.from_numpy(x[z + 1]).unsqueeze(0).float())
        y_center = self._resize_slice(torch.from_numpy(y[z]).unsqueeze(0).float())
        mask_center = self._resize_slice(torch.from_numpy(mask[z]).unsqueeze(0).float(), is_mask=True)
        if self.return_meta:
            return {"x_prev": x_prev, "x_center": x_center, "x_next": x_next, "y_center": y_center, "mask_center": mask_center, "case_id": case_id, "slice_idx": z}
        return x_prev, x_center, x_next, y_center, mask_center
