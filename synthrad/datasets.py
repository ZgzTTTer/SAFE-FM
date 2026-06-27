from pathlib import Path
import sys

from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets_mask import SynthRAD2p5DDataset as SharedSynthRAD2p5DDataset


class SynthRAD2p5DDataset(Dataset):
    def __init__(
        self,
        root,
        task="task1",
        anatomy="HN",
        split="train",
        radius=1,
        target_size=(192, 256),
        cache_volumes=True,
        ct_clip=(-1000.0, 2000.0),
        return_meta=False,
        keep_ratio=1.0,
        mask_margin=10,
    ):
        super().__init__()
        self.dataset = SharedSynthRAD2p5DDataset(
            root=root,
            task=task,
            anatomy=anatomy,
            split=split,
            radius=radius,
            target_size=target_size,
            cache_volumes=cache_volumes,
            ct_clip=ct_clip,
            return_meta=True,
            keep_ratio=float(keep_ratio),
            mask_margin=mask_margin,
        )
        self.case_infos = self.dataset.case_infos
        self.samples = self.dataset.samples
        self.return_meta = return_meta

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        x_prev = sample["x_prev"]
        x_center = sample["x_center"]
        x_next = sample["x_next"]
        y_center = sample["y_center"]
        mask_center = sample["mask_center"]
        if self.return_meta:
            return {
                "x_prev": x_prev,
                "x_center": x_center,
                "x_next": x_next,
                "y_center": y_center,
                "mask": mask_center,
                "case_id": sample["case_id"],
                "slice_idx": sample["slice_idx"],
            }
        return x_prev, x_center, x_next, y_center, mask_center
