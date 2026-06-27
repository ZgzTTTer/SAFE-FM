import argparse

import numpy as np
import torch

from train import train


def build_args():
    parser = argparse.ArgumentParser(description="Conditional Flow Matching - 2.5D Medical Image Translation")

    parser.add_argument("--data_root", type=str, default="../datasets/SynthRAD2025")
    parser.add_argument("--task", type=str, default="task1", choices=["task1", "task2"])
    parser.add_argument("--anatomy", type=str, default="HN", choices=["AB", "HN", "TH"])

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--use_mask", type=lambda x: str(x).lower() != "false", default=True,
                        help="whether to apply mask in training loss and metrics")
    parser.add_argument("--target_h", type=int, default=192)
    parser.add_argument("--target_w", type=int, default=256)
    parser.add_argument("--keep_ratio", type=float, default=0.2)
    parser.add_argument("--mask_margin", type=int, default=10)

    parser.add_argument("--cache_volumes", default=True, action="store_true")
    parser.add_argument("--no_cache_volumes", action="store_true")

    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--channel_mults", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--time_emb_dim", type=int, default=256)

    parser.add_argument("--fm_method", type=str, default="cfm",
                        choices=["cfm", "otcfm", "target", "sbcfm", "vpcfm"],
                        help="Flow matching method: cfm | otcfm | target | sbcfm | vpcfm")
    parser.add_argument("--sigma", type=float, default=0.0,
                        help="Sigma for the flow matcher (noise level)")
    parser.add_argument("--sigma_min", type=float, default=1e-4)

    parser.add_argument("--use_feature_fusion", action="store_true", default=True,
                        help="Enable Feature-level Adaptive Fusion")
    parser.add_argument("--no_feature_fusion", action="store_true",
                        help="Disable Feature-level Adaptive Fusion")
    parser.add_argument("--fusion_feat_channels", type=int, default=32)

    parser.add_argument("--use_asa_fm", action="store_true", default=True,
                        help="Enable Anatomical Source-Anchored Flow Matching")
    parser.add_argument("--no_asa_fm", action="store_true",
                        help="Disable ASA-FM")
    parser.add_argument("--asa_alpha", type=float, default=0.5,
                        help="ASA-FM mixing coefficient")

    parser.add_argument("--use_boundary_reg", action="store_true", default=True,
                        help="Enable Boundary-Aware Velocity Gradient Regularization")
    parser.add_argument("--no_boundary_reg", action="store_true",
                        help="Disable Boundary Regularization")
    parser.add_argument("--boundary_lambda", type=float, default=0.1,
                        help="Boundary regularization weight")

    parser.add_argument("--val_ssim_start", type=int, default=60)
    parser.add_argument("--val_ssim_interval", type=int, default=10)
    parser.add_argument("--val_sample_steps", type=int, default=50)
    parser.add_argument("--num_vis_samples", type=int, default=1)
    parser.add_argument("--val_solver", type=str, default="euler", choices=["euler", "heun"],
                        help="ODE solver used during validation sampling")
    parser.add_argument("--use_ema", type=lambda x: str(x).lower() != "false", default=True,
                        help="whether to maintain EMA weights during training")
    parser.add_argument("--ema_decay", type=float, default=0.999)

    parser.add_argument("--run_name", type=str, default="",
                        help="Experiment name. Auto-generated if empty.")
    parser.add_argument("--save_dir", type=str, default="./outputs/sensitivity")
    parser.add_argument("--log_dir", type=str, default="./logs/sensitivity")
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=50)

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--resume", type=str, default=None)

    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = build_args()
    if args.no_feature_fusion:
        args.use_feature_fusion = False
    if args.no_asa_fm:
        args.use_asa_fm = False
    if args.no_boundary_reg:
        args.use_boundary_reg = False
    set_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
