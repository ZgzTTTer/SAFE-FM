import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_INTERACTIVE = sys.stdout.isatty()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from synthrad_slice_export import FixedSliceExporter

from datasets import SynthRAD2p5DDataset
from network import ConditionUNet
from synthrad_fm import SynthRADFlowMatcher
from utils.metrics import (
    compute_batch_metrics, compute_fid,
    _extract_inception_features, _FID_AVAILABLE,
)
from utils.visualize import save_sample_panel


def build_args():
    parser = argparse.ArgumentParser(description="SAFE-CFM Test / Inference")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pth checkpoint")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--task", type=str, default="task1", choices=["task1", "task2"])
    parser.add_argument("--anatomy", type=str, default="HN", choices=["AB", "HN", "TH"])
    parser.add_argument("--split", type=str, default="val",
                        help="Which split to evaluate: train | val | test")

    parser.add_argument("--fm_method", type=str, default=None,
                        help="Override FM method (auto-detected from checkpoint if not set)")
    parser.add_argument("--sigma", type=float, default=0.0)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of ODE solver steps")
    parser.add_argument("--solver", type=str, default="euler",
                        choices=["euler", "heun"])

    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--use_mask", type=lambda x: str(x).lower() != "false", default=True,
                        help="whether to apply mask in testing metrics")
    parser.add_argument("--target_h", type=int, default=192)
    parser.add_argument("--target_w", type=int, default=256)
    parser.add_argument("--keep_ratio", type=float, default=0.2)
    parser.add_argument("--mask_margin", type=int, default=10)

    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--channel_mults", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--time_emb_dim", type=int, default=256)

    parser.add_argument("--use_feature_fusion", action="store_true", default=True)
    parser.add_argument("--no_feature_fusion", action="store_true")
    parser.add_argument("--fusion_feat_channels", type=int, default=32)

    parser.add_argument("--use_asa_fm", action="store_true", default=True)
    parser.add_argument("--no_asa_fm", action="store_true")
    parser.add_argument("--asa_alpha", type=float, default=0.5)

    parser.add_argument("--use_boundary_reg", action="store_true", default=True)
    parser.add_argument("--no_boundary_reg", action="store_true")
    parser.add_argument("--boundary_lambda", type=float, default=0.1)

    parser.add_argument("--use_ema", type=lambda x: str(x).lower() != "false", default=True,
                        help="whether to maintain EMA weights during training")
    parser.add_argument("--ema_decay", type=float, default=0.999)

    parser.add_argument("--save_images", action="store_true",
                        help="Save predicted images")
    parser.add_argument("--export_slice_ratios", type=float, nargs="+", default=[0.3, 0.7],
                        help="Fixed percentile slice positions per case for PNG export")
    parser.add_argument("--output_dir", type=str, default="./test_results")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)

    return parser.parse_args()


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


def count_parameters_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def test(args):
    device = torch.device(args.device)
    print(f"Device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}")
    if "metrics" in checkpoint:
        print(f"  Train metrics: {checkpoint['metrics']}")

    fm_method = args.fm_method
    if fm_method is None:
        fm_method = checkpoint.get("fm_method", "cfm")
    print(f"  FM method: {fm_method}")

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
    ).to(device)

    trainer = SynthRADFlowMatcher(
        model=model,
        fm_method=fm_method,
        sigma=args.sigma,
        device=device,
        use_feature_fusion=args.use_feature_fusion,
        use_asa_fm=args.use_asa_fm,
        use_boundary_reg=args.use_boundary_reg,
        asa_alpha=args.asa_alpha,
        boundary_lambda=args.boundary_lambda,
        fusion_feat_channels=args.fusion_feat_channels,
        use_mask=args.use_mask,
    )

    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    if "fusion_state_dict" in checkpoint and trainer.fusion_module is not None:
        trainer.fusion_module.load_state_dict(checkpoint["fusion_state_dict"])
    print("Model weights loaded.")
    print(f"Model parameters: {count_parameters_m(trainer.model):.2f} M")

    target_size = (args.target_h, args.target_w)
    return_meta = args.save_images
    dataset = SynthRAD2p5DDataset(
        root=args.data_root, task=args.task, anatomy=args.anatomy,
        split=args.split, radius=args.radius, target_size=target_size,
        cache_volumes=True, return_meta=return_meta,
        keep_ratio=args.keep_ratio, mask_margin=args.mask_margin,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Dataset: split={args.split}, samples={len(dataset)}")

    sample_fn = trainer.sample_heun if args.solver == "heun" else trainer.sample
    print(f"Solver: {args.solver}, steps={args.num_steps}")

    if args.save_images:
        out_dir = Path(args.output_dir) / f"{fm_method}_{args.task}_{args.anatomy}"
        out_dir.mkdir(parents=True, exist_ok=True)
        selected_png_dir = out_dir / "selected_png"
        selected_png_dir.mkdir(parents=True, exist_ok=True)
        slice_exporter = FixedSliceExporter(selected_png_dir, dataset.case_infos, ratios=args.export_slice_ratios)

    trainer.eval()
    all_ssim, all_psnr, all_mae, all_nmse, all_pcc, all_lpips = [], [], [], [], [], []
    all_pred_feats, all_target_feats = [], []

    t_start = time.time()
    for batch_idx, batch_data in enumerate(tqdm(loader, desc="Testing", disable=not _INTERACTIVE)):
        if return_meta:
            condition, y_center, mask = _unpack_batch(batch_data, device)
        else:
            condition, y_center, mask = _unpack_batch(batch_data, device)

        with torch.no_grad():
            y_pred = sample_fn(condition, num_steps=args.num_steps)

        metric_mask = mask if args.use_mask else None
        m = compute_batch_metrics(y_pred, y_center, mask=metric_mask, device=device)
        all_ssim.append(m["ssim"])
        all_psnr.append(m["psnr"])
        all_mae.append(m["mae"])
        all_nmse.append(m["nmse"])
        all_pcc.append(m["pcc"])
        if m["lpips"] is not None:
            all_lpips.append(m["lpips"])

        if _FID_AVAILABLE:
            all_pred_feats.append(_extract_inception_features(y_pred, device))
            all_target_feats.append(_extract_inception_features(y_center, device))

        if args.save_images and return_meta:
            for i in range(y_pred.shape[0]):
                case_id = batch_data["case_id"][i]
                slice_idx = int(batch_data["slice_idx"][i])
                sample_dir = out_dir / f"{case_id}_z{slice_idx:03d}"
                save_sample_panel(
                    x_prev=condition[i, 0:1], x_center=condition[i, 1:2],
                    x_next=condition[i, 2:3], y_true=y_center[i], y_pred=y_pred[i],
                    save_dir=sample_dir, task=args.task,
                )
                slice_exporter.save(
                    case_id=str(case_id),
                    slice_idx=slice_idx,
                    condition=condition[i, 1:2],
                    target=y_center[i],
                    prediction=y_pred[i],
                )

    elapsed = time.time() - t_start
    n = len(all_ssim)

    metrics = {
        "ssim": sum(all_ssim) / n,
        "psnr": sum(all_psnr) / n,
        "mae": sum(all_mae) / n,
        "nmse": sum(all_nmse) / n,
        "pcc": sum(all_pcc) / n,
    }
    if all_lpips:
        metrics["lpips"] = sum(all_lpips) / len(all_lpips)
    else:
        metrics["lpips"] = None
    if _FID_AVAILABLE and all_pred_feats:
        pred_feats = np.concatenate(all_pred_feats, axis=0)
        target_feats = np.concatenate(all_target_feats, axis=0)
        metrics["fid"] = compute_fid(pred_feats, target_feats)
    else:
        metrics["fid"] = None

    print(f"\n{'='*50}")
    print(f"  Test Results ({args.split} split)")
    print(f"  FM: {fm_method} | Solver: {args.solver} | Steps: {args.num_steps}")
    print(f"{'='*50}")
    print(f"  SSIM:  {metrics['ssim']:.4f}")
    print(f"  PSNR:  {metrics['psnr']:.2f} dB")
    print(f"  MAE:   {metrics['mae']:.4f}")
    print(f"  NMSE:  {metrics['nmse']:.6f}")
    print(f"  PCC:   {metrics['pcc']:.4f}")
    if metrics.get("lpips") is not None:
        print(f"  LPIPS: {metrics['lpips']:.4f}")
    if metrics.get("fid") is not None:
        print(f"  FID:   {metrics['fid']:.2f}")
    print(f"{'='*50}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(dataset):.3f}s/sample)")
    if args.save_images:
        print(f"  Selected PNG summary: {slice_exporter.summary()}")

    return metrics


if __name__ == "__main__":
    args = build_args()
    if args.no_feature_fusion:
        args.use_feature_fusion = False
    if args.no_asa_fm:
        args.use_asa_fm = False
    if args.no_boundary_reg:
        args.use_boundary_reg = False

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    test(args)
