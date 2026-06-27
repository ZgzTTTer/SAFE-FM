from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def tensor_to_numpy_img(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    return x


def minus1_1_to_01(img):
    img = np.clip(img, -1.0, 1.0)
    return np.clip((img + 1.0) / 2.0, 0.0, 1.0)


def tensor_to_display_img(x):
    return minus1_1_to_01(tensor_to_numpy_img(x))


def save_gray_image(img, save_path, dpi=600):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig_size = 256.0 / dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_sample_panel(
    x_prev, x_center, x_next,
    y_true=None, y_pred=None,
    save_dir=None, save_path=None,
    task="task1",
    titles=None, dpi=300,
):
    input_name = "MRI" if task == "task1" else "CBCT"
    items = [
        (tensor_to_display_img(x_prev), f"{input_name}_z-1"),
        (tensor_to_display_img(x_center), f"{input_name}_z"),
        (tensor_to_display_img(x_next), f"{input_name}_z+1"),
    ]
    if y_true is not None:
        items.append((tensor_to_display_img(y_true), "CT_gt"))
    if y_pred is not None:
        items.append((tensor_to_display_img(y_pred), "CT_pred"))
    if titles is not None:
        items = [(img, t) for (img, _), t in zip(items, titles)]

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for img, title in items:
            save_gray_image(img, save_dir / f"{title}.png", dpi=dpi)
        return None

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        n = len(items)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
        if n == 1:
            axes = [axes]
        for ax, (img, title) in zip(axes, items):
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(title)
            ax.axis("off")
        plt.tight_layout()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return None

    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, (img, title) in zip(axes, items):
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    return fig
