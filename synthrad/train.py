import logging
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

_INTERACTIVE = sys.stdout.isatty()

from datasets import SynthRAD2p5DDataset
from network import ConditionUNet
from synthrad_fm import SynthRADFlowMatcher
from utils.metrics import (
    compute_batch_metrics, compute_fid,
    _extract_inception_features, _FID_AVAILABLE,
)
from utils.visualize import save_sample_panel


class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_params = dict(self.shadow.named_parameters())
        model_params = dict(model.named_parameters())
        for name, param in model_params.items():
            ema_params[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

        ema_buffers = dict(self.shadow.named_buffers())
        model_buffers = dict(model.named_buffers())
        for name, buf in model_buffers.items():
            ema_buffers[name].copy_(buf.detach())

    def to(self, device):
        self.shadow.to(device)
        return self

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


def create_dataloaders(args):
    cache = not args.no_cache_volumes
    target_size = (args.target_h, args.target_w)

    train_ds = SynthRAD2p5DDataset(
        root=args.data_root, task=args.task, anatomy=args.anatomy,
        split="train", radius=args.radius, target_size=target_size,
        cache_volumes=cache, return_meta=False, keep_ratio=args.keep_ratio,
        mask_margin=args.mask_margin,
    )
    val_ds = SynthRAD2p5DDataset(
        root=args.data_root, task=args.task, anatomy=args.anatomy,
        split="val", radius=args.radius, target_size=target_size,
        cache_volumes=cache, return_meta=False, keep_ratio=args.keep_ratio,
        mask_margin=args.mask_margin,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def _unpack_batch(batch_data, device):
    if isinstance(batch_data, dict):
        x_prev = batch_data["x_prev"].to(device, non_blocking=True)
        x_center = batch_data["x_center"].to(device, non_blocking=True)
        x_next = batch_data["x_next"].to(device, non_blocking=True)
        y_center = batch_data["y_center"].to(device, non_blocking=True)
        mask = batch_data.get("mask")
        if mask is None:
            mask = batch_data.get("mask_center")
        if mask is not None:
            mask = mask.to(device, non_blocking=True)
    else:
        if len(batch_data) == 5:
            x_prev, x_center, x_next, y_center, mask = batch_data
            mask = mask.to(device, non_blocking=True)
        else:
            x_prev, x_center, x_next, y_center = batch_data
            mask = None
        x_prev = x_prev.to(device, non_blocking=True)
        x_center = x_center.to(device, non_blocking=True)
        x_next = x_next.to(device, non_blocking=True)
        y_center = y_center.to(device, non_blocking=True)
    condition = torch.cat([x_prev, x_center, x_next], dim=1)
    return condition, y_center, mask


def train_one_epoch(trainer, train_loader, optimizer, device, epoch, writer, log_interval, grad_clip=1.0, ema_model=None, ema_fusion=None):
    trainer.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not _INTERACTIVE)
    for batch_idx, batch_data in enumerate(pbar):
        condition, y_center, mask = _unpack_batch(batch_data, device)
        optimizer.zero_grad()
        loss = trainer.compute_loss(y_center, condition, mask=mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.get_all_parameters(), grad_clip)
        optimizer.step()
        if ema_model is not None:
            ema_model.update(trainer.model)
        if ema_fusion is not None and trainer.fusion_module is not None:
            ema_fusion.update(trainer.fusion_module)

        total_loss += loss.item()
        global_step = epoch * num_batches + batch_idx
        if batch_idx % log_interval == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / num_batches


@torch.no_grad()
def validate_loss(trainer, val_loader, device):
    trainer.eval()
    total_loss = 0.0
    num_batches = len(val_loader)
    for batch_data in val_loader:
        condition, y_center, mask = _unpack_batch(batch_data, device)
        loss = trainer.compute_loss(y_center, condition, mask=mask)
        total_loss += loss.item()
    return total_loss / num_batches


def _with_ema_modules(trainer, ema_model=None, ema_fusion=None):
    class _EMAScope:
        def __enter__(self_inner):
            self_inner.model = trainer.model
            self_inner.fusion_module = trainer.fusion_module
            if ema_model is not None:
                trainer.model = ema_model.shadow
            if ema_fusion is not None:
                trainer.fusion_module = ema_fusion.shadow
            return trainer

        def __exit__(self_inner, exc_type, exc, tb):
            trainer.model = self_inner.model
            trainer.fusion_module = self_inner.fusion_module
            return False

    return _EMAScope()


@torch.no_grad()
def validate_metrics(trainer, val_loader, device, num_sample_steps=50, solver="euler"):
    trainer.eval()
    all_ssim, all_psnr, all_mae, all_lpips = [], [], [], []
    all_pred_feats, all_target_feats = [], []

    sample_fn = trainer.sample_heun if solver == "heun" else trainer.sample

    for batch_data in tqdm(val_loader, desc="Val Metrics", disable=not _INTERACTIVE):
        condition, y_center, mask = _unpack_batch(batch_data, device)
        y_pred = sample_fn(condition, num_steps=num_sample_steps)

        m = compute_batch_metrics(y_pred, y_center, mask=mask, device=device)
        all_ssim.append(m["ssim"])
        all_psnr.append(m["psnr"])
        all_mae.append(m["mae"])
        if m["lpips"] is not None:
            all_lpips.append(m["lpips"])

        if _FID_AVAILABLE:
            all_pred_feats.append(_extract_inception_features(y_pred, device))
            all_target_feats.append(_extract_inception_features(y_center, device))

    n = len(all_ssim)
    result = {
        "ssim": sum(all_ssim) / n,
        "psnr": sum(all_psnr) / n,
        "mae": sum(all_mae) / n,
    }
    if all_lpips:
        result["lpips"] = sum(all_lpips) / len(all_lpips)
    else:
        result["lpips"] = None
    if _FID_AVAILABLE and all_pred_feats:
        pred_feats = np.concatenate(all_pred_feats, axis=0)
        target_feats = np.concatenate(all_target_feats, axis=0)
        result["fid"] = compute_fid(pred_feats, target_feats)
    else:
        result["fid"] = None
    return result


def _build_checkpoint_state(trainer, optimizer, epoch, metrics, ema_model=None, ema_fusion=None, use_ema_weights=False):
    model_state = ema_model.state_dict() if use_ema_weights and ema_model is not None else trainer.model.state_dict()
    state = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics,
        "fm_method": trainer.fm_method,
        "is_ema": bool(use_ema_weights),
    }
    if trainer.fusion_module is not None:
        if use_ema_weights and ema_fusion is not None:
            state["fusion_state_dict"] = ema_fusion.state_dict()
        else:
            state["fusion_state_dict"] = trainer.fusion_module.state_dict()
    if ema_model is not None and not use_ema_weights:
        state["ema_model_state_dict"] = ema_model.state_dict()
    if ema_fusion is not None and not use_ema_weights:
        state["ema_fusion_state_dict"] = ema_fusion.state_dict()
    return state


def save_checkpoint(trainer, optimizer, epoch, metrics, save_path, ema_model=None, ema_fusion=None, use_ema_weights=False):
    torch.save(
        _build_checkpoint_state(trainer, optimizer, epoch, metrics, ema_model, ema_fusion, use_ema_weights),
        save_path,
    )


def save_checkpoint_set(trainer, optimizer, epoch, metrics, save_dir, ema_model=None, ema_fusion=None, save_last=False, save_ema_last=False, save_ema_best_ssim=False, save_ema_best_psnr=False):
    if save_last:
        save_checkpoint(trainer, optimizer, epoch, metrics, save_dir / "last.pth", ema_model=ema_model, ema_fusion=ema_fusion, use_ema_weights=False)
    if save_ema_last:
        save_checkpoint(trainer, None, epoch, metrics, save_dir / "ema_last.pth", ema_model=ema_model, ema_fusion=ema_fusion, use_ema_weights=True)
    if save_ema_best_ssim:
        save_checkpoint(trainer, None, epoch, metrics, save_dir / "ema_best_ssim.pth", ema_model=ema_model, ema_fusion=ema_fusion, use_ema_weights=True)
    if save_ema_best_psnr:
        save_checkpoint(trainer, None, epoch, metrics, save_dir / "ema_best_psnr.pth", ema_model=ema_model, ema_fusion=ema_fusion, use_ema_weights=True)


def load_checkpoint(trainer, optimizer, checkpoint_path, ema_model=None, ema_fusion=None):
    checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    if "fusion_state_dict" in checkpoint and trainer.fusion_module is not None:
        trainer.fusion_module.load_state_dict(checkpoint["fusion_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if ema_model is not None:
        if "ema_model_state_dict" in checkpoint:
            ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
        elif checkpoint.get("is_ema", False):
            ema_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            ema_model.load_state_dict(trainer.model.state_dict())
    if ema_fusion is not None and trainer.fusion_module is not None:
        if "ema_fusion_state_dict" in checkpoint:
            ema_fusion.load_state_dict(checkpoint["ema_fusion_state_dict"])
        elif checkpoint.get("is_ema", False) and "fusion_state_dict" in checkpoint:
            ema_fusion.load_state_dict(checkpoint["fusion_state_dict"])
        else:
            ema_fusion.load_state_dict(trainer.fusion_module.state_dict())
    return checkpoint["epoch"], checkpoint.get("metrics", {})


def _build_vis_indices(meta_dataset, num_per_case):
    from collections import defaultdict
    case_to_indices = defaultdict(list)
    for idx, (case_id, _z) in enumerate(meta_dataset.samples):
        case_to_indices[case_id].append(idx)
    selected = []
    for case_id in sorted(case_to_indices.keys()):
        indices = case_to_indices[case_id]
        n = len(indices)
        if n <= num_per_case:
            picks = indices
        else:
            step_positions = np.linspace(0, n - 1, num_per_case, dtype=int)
            picks = [indices[p] for p in step_positions]
        selected.extend(picks)
    return selected


def save_val_samples(trainer, meta_dataset, device, save_dir, epoch, task, num_samples=1, num_steps=50, solver="euler"):
    trainer.eval()
    vis_indices = _build_vis_indices(meta_dataset, num_samples)
    vis_subset = Subset(meta_dataset, vis_indices)
    vis_loader = DataLoader(vis_subset, batch_size=1, shuffle=False, num_workers=0)

    vis_dir = Path(save_dir) / f"epoch_{epoch:04d}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    sample_fn = trainer.sample_heun if solver == "heun" else trainer.sample

    for batch_data in vis_loader:
        case_id = batch_data["case_id"][0]
        slice_idx = int(batch_data["slice_idx"][0])
        condition, y_center, _mask = _unpack_batch(batch_data, device)

        with torch.no_grad():
            y_pred = sample_fn(condition, num_steps=num_steps)

        sample_dir = vis_dir / f"{case_id}_z{slice_idx:03d}"
        save_sample_panel(
            x_prev=condition[0, 0:1], x_center=condition[0, 1:2],
            x_next=condition[0, 2:3], y_true=y_center[0], y_pred=y_pred[0],
            save_dir=sample_dir, task=task,
        )


def _get_run_name(args):
    if args.run_name:
        return args.run_name
    parts = [args.fm_method]
    if args.use_feature_fusion:
        parts.append("ff")
    if args.use_asa_fm:
        parts.append("asa")
    if args.use_boundary_reg:
        parts.append("br")
    parts.extend([args.task, args.anatomy])
    return "_".join(parts)


def _setup_text_logger(log_dir, run_name):
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(Path(log_dir) / "training.log", mode="a")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def train(args):
    args.device = torch.device(args.device)
    print(f"Using device: {args.device}")
    if args.device.type == "cuda":
        torch.cuda.set_device(args.device)
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(args.device)}")

    run_name = _get_run_name(args)
    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = save_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir)
    txt_logger = _setup_text_logger(log_dir, run_name)

    args_path = log_dir / "args.txt"
    with open(args_path, "w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")

    print(f"Run: {run_name}, Save: {save_dir}, Log: {log_dir}")
    txt_logger.info(f"=== Run: {run_name} ===")

    train_loader, val_loader = create_dataloaders(args)
    target_size = (args.target_h, args.target_w)
    val_meta_dataset = SynthRAD2p5DDataset(
        root=args.data_root, task=args.task, anatomy=args.anatomy,
        split="val", radius=args.radius, target_size=target_size,
        cache_volumes=not args.no_cache_volumes, return_meta=True,
        keep_ratio=args.keep_ratio, mask_margin=args.mask_margin,
    )

    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    if args.use_feature_fusion:
        in_ch = 1 + args.fusion_feat_channels
    else:
        in_ch = 4

    model = ConditionUNet(
        in_channels=in_ch, out_channels=1,
        base_channels=args.base_channels,
        channel_mults=tuple(args.channel_mults),
        num_res_blocks=args.num_res_blocks,
        time_emb_dim=args.time_emb_dim,
    ).to(args.device)

    trainer = SynthRADFlowMatcher(
        model=model,
        fm_method=args.fm_method,
        sigma=args.sigma,
        device=args.device,
        use_feature_fusion=args.use_feature_fusion,
        use_asa_fm=args.use_asa_fm,
        use_boundary_reg=args.use_boundary_reg,
        asa_alpha=args.asa_alpha,
        boundary_lambda=args.boundary_lambda,
        fusion_feat_channels=args.fusion_feat_channels,
        use_mask=args.use_mask,
    )

    all_params = trainer.get_all_parameters()
    num_params = sum(p.numel() for p in all_params if p.requires_grad)
    print(f"Model params: {num_params:,}")
    print(f"  fm_method={args.fm_method}, sigma={args.sigma}")
    print(f"  feature_fusion={args.use_feature_fusion}, asa_fm={args.use_asa_fm}, boundary_reg={args.use_boundary_reg}")

    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    ema_model = EMAModel(trainer.model, decay=args.ema_decay).to(args.device) if args.use_ema else None
    ema_fusion = EMAModel(trainer.fusion_module, decay=args.ema_decay).to(args.device) if args.use_ema and trainer.fusion_module is not None else None
    if args.use_ema:
        print(f"  EMA enabled: decay={args.ema_decay}")

    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        start_epoch, _ = load_checkpoint(trainer, optimizer, args.resume, ema_model=ema_model, ema_fusion=ema_fusion)
        start_epoch += 1

    best_ssim = -1.0
    best_psnr = -1.0

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            trainer, train_loader, optimizer, args.device,
            epoch, writer, args.log_interval, args.grad_clip,
            ema_model=ema_model, ema_fusion=ema_fusion,
        )
        with _with_ema_modules(trainer, ema_model if args.use_ema else None, ema_fusion if args.use_ema else None):
            val_loss = validate_loss(trainer, val_loader, args.device)
        epoch_time = time.time() - epoch_start

        msg = f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, time={epoch_time:.1f}s"
        print(msg, flush=True)
        txt_logger.info(msg)
        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        writer.add_scalar("epoch/val_loss", val_loss, epoch)

        if (epoch + 1) >= args.val_ssim_start and (epoch + 1) % args.val_ssim_interval == 0:
            with _with_ema_modules(trainer, ema_model if args.use_ema else None, ema_fusion if args.use_ema else None):
                metrics = validate_metrics(trainer, val_loader, args.device, args.val_sample_steps, args.val_solver)
            parts = [f"ssim={metrics['ssim']:.4f}", f"psnr={metrics['psnr']:.2f}", f"mae={metrics['mae']:.4f}"]
            if metrics["lpips"] is not None:
                parts.append(f"lpips={metrics['lpips']:.4f}")
            if metrics["fid"] is not None:
                parts.append(f"fid={metrics['fid']:.2f}")
            val_msg = f"  Val: solver={args.val_solver}, steps={args.val_sample_steps}, {', '.join(parts)}"
            print(val_msg, flush=True)
            txt_logger.info(val_msg)

            writer.add_scalar("epoch/val_ssim", metrics["ssim"], epoch)
            writer.add_scalar("epoch/val_psnr", metrics["psnr"], epoch)
            writer.add_scalar("epoch/val_mae", metrics["mae"], epoch)

            if metrics["ssim"] > best_ssim:
                best_ssim = metrics["ssim"]
                save_checkpoint_set(
                    trainer, optimizer, epoch, {"val_loss": val_loss, **metrics}, save_dir,
                    ema_model=ema_model, ema_fusion=ema_fusion, save_ema_best_ssim=True,
                )
                print(f"  Saved EMA best SSIM={best_ssim:.4f}", flush=True)
                txt_logger.info(f"  Saved EMA best SSIM={best_ssim:.4f}")

            if metrics["psnr"] > best_psnr:
                best_psnr = metrics["psnr"]
                save_checkpoint_set(
                    trainer, optimizer, epoch, {"val_loss": val_loss, **metrics}, save_dir,
                    ema_model=ema_model, ema_fusion=ema_fusion, save_ema_best_psnr=True,
                )
                print(f"  Saved EMA best PSNR={best_psnr:.2f}", flush=True)
                txt_logger.info(f"  Saved EMA best PSNR={best_psnr:.2f}")

            with _with_ema_modules(trainer, ema_model if args.use_ema else None, ema_fusion if args.use_ema else None):
                save_val_samples(trainer, val_meta_dataset, args.device, vis_dir, epoch, args.task, args.num_vis_samples, args.val_sample_steps, args.val_solver)

        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint_set(
                trainer, optimizer, epoch, {"val_loss": val_loss}, save_dir,
                ema_model=ema_model, ema_fusion=ema_fusion, save_last=True, save_ema_last=True,
            )

    save_checkpoint_set(
        trainer, optimizer, args.epochs - 1, {"val_loss": val_loss}, save_dir,
        ema_model=ema_model, ema_fusion=ema_fusion, save_last=True, save_ema_last=True,
    )
    writer.close()
    txt_logger.info(f"Done. Best EMA SSIM: {best_ssim:.4f}, Best EMA PSNR: {best_psnr:.2f}")
    print(f"Done. Best EMA SSIM: {best_ssim:.4f}, Best EMA PSNR: {best_psnr:.2f}")
