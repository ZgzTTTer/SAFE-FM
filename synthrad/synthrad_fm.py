import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from feature_fusion import FeatureAdaptiveFusion


FM_REGISTRY = {
    "cfm": ConditionalFlowMatcher,
    "otcfm": ExactOptimalTransportConditionalFlowMatcher,
    "target": TargetConditionalFlowMatcher,
    "sbcfm": SchrodingerBridgeConditionalFlowMatcher,
    "vpcfm": VariancePreservingConditionalFlowMatcher,
}


def build_flow_matcher(method="cfm", sigma=0.0):
    if method not in FM_REGISTRY:
        raise ValueError(f"Unknown FM method '{method}'. Choose from {list(FM_REGISTRY.keys())}")
    cls = FM_REGISTRY[method]
    if method == "sbcfm":
        return cls(sigma=max(sigma, 1e-3))
    return cls(sigma=sigma)


class SobelGradient2D(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, x):
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        return grad_x, grad_y


class SynthRADFlowMatcher:
    def __init__(
        self,
        model,
        fm_method="cfm",
        sigma=0.0,
        device="cuda",
        use_feature_fusion=True,
        use_asa_fm=True,
        use_boundary_reg=True,
        asa_alpha=0.5,
        boundary_lambda=0.1,
        fusion_feat_channels=32,
        use_mask=True,
    ):
        self.model = model
        self.device = device
        self.fm = build_flow_matcher(fm_method, sigma)
        self.fm_method = fm_method

        self.use_feature_fusion = use_feature_fusion
        self.use_asa_fm = use_asa_fm
        self.use_boundary_reg = use_boundary_reg
        self.asa_alpha = asa_alpha
        self.boundary_lambda = boundary_lambda
        self.use_mask = use_mask

        if self.use_feature_fusion:
            self.fusion_module = FeatureAdaptiveFusion(
                in_channels=1,
                feat_channels=fusion_feat_channels,
                num_slices=3,
            ).to(device)
        else:
            self.fusion_module = None

        if self.use_boundary_reg:
            self.sobel = SobelGradient2D().to(device)
        else:
            self.sobel = None

    def get_all_parameters(self):
        params = list(self.model.parameters())
        if self.fusion_module is not None:
            params += list(self.fusion_module.parameters())
        return params

    def train(self):
        self.model.train()
        if self.fusion_module is not None:
            self.fusion_module.train()

    def eval(self):
        self.model.eval()
        if self.fusion_module is not None:
            self.fusion_module.eval()

    def _fuse_condition(self, condition):
        if not self.use_feature_fusion or self.fusion_module is None:
            return condition
        x_prev = condition[:, 0:1, :, :]
        x_center = condition[:, 1:2, :, :]
        x_next = condition[:, 2:3, :, :]
        return self.fusion_module([x_prev, x_center, x_next])

    def _build_base_sample(self, x_1, cond_z):
        if self.use_asa_fm:
            alpha = self.asa_alpha
            eps = torch.randn_like(x_1)
            x_0 = (alpha ** 0.5) * cond_z + ((1 - alpha) ** 0.5) * eps
        else:
            x_0 = torch.randn_like(x_1)
        return x_0

    def compute_loss(self, x_1, condition, mask=None):
        batch_size = x_1.shape[0]

        cond_fused = self._fuse_condition(condition)
        cond_z = condition[:, 1:2, :, :]

        x_0 = self._build_base_sample(x_1, cond_z)

        t, x_t, u_t = self.fm.sample_location_and_conditional_flow(x_0, x_1)

        if self.use_feature_fusion:
            model_input = torch.cat([x_t, cond_fused], dim=1)
        else:
            model_input = torch.cat([x_t, condition], dim=1)

        predicted_u_t = self.model(model_input, t)

        if self.use_mask and mask is not None:
            mask = (mask > 0.5).float().expand_as(predicted_u_t)
            denom = mask.sum().clamp_min(1.0)
            mse_loss = ((predicted_u_t - u_t) ** 2 * mask).sum() / denom
        else:
            mse_loss = F.mse_loss(predicted_u_t, u_t, reduction='mean')

        if self.use_boundary_reg and self.sobel is not None:
            pred_gx, pred_gy = self.sobel(predicted_u_t)
            gt_gx, gt_gy = self.sobel(u_t)
            if self.use_mask and mask is not None:
                grad_mask = mask.expand_as(pred_gx)
                grad_denom = grad_mask.sum().clamp_min(1.0)
                grad_loss = (((pred_gx - gt_gx) ** 2) * grad_mask).sum() / grad_denom + \
                            (((pred_gy - gt_gy) ** 2) * grad_mask).sum() / grad_denom
            else:
                grad_loss = F.mse_loss(pred_gx, gt_gx, reduction='mean') + \
                            F.mse_loss(pred_gy, gt_gy, reduction='mean')
            total_loss = mse_loss + self.boundary_lambda * grad_loss
        else:
            total_loss = mse_loss

        return total_loss

    @torch.no_grad()
    def sample(self, condition, shape=None, num_steps=50, eps_noise=None, debug=False):
        batch_size = condition.shape[0]
        if shape is None:
            shape = (batch_size, 1, condition.shape[2], condition.shape[3])
    
        cond_fused = self._fuse_condition(condition)
        cond_z = condition[:, 1:2, :, :]
    
        # 固定或外部传入初始噪声，方便不同 N 公平比较
        if eps_noise is None:
            eps_noise = torch.randn(shape, device=self.device)
    
        if self.use_asa_fm:
            alpha = self.asa_alpha
            x = (alpha ** 0.5) * cond_z + ((1 - alpha) ** 0.5) * eps_noise
        else:
            x = eps_noise
    
        def check_tensor(name, tensor, step, t):
            if not torch.isfinite(tensor).all():
                print(f"\n[ERROR] {name} has NaN or Inf")
                print("step:", step)
                print("t:", float(t[0].item()))
                print("shape:", tuple(tensor.shape))
                print("min:", torch.nan_to_num(tensor).min().item())
                print("max:", torch.nan_to_num(tensor).max().item())
                print("mean:", torch.nan_to_num(tensor).mean().item())
                print("max_abs:", torch.nan_to_num(tensor).abs().max().item())
                raise RuntimeError(f"{name} is not finite")
    
        eps_t = 1e-4
        dt = 1.0 / num_steps
    
        for i in range(num_steps):
            # 用 midpoint time，避开精确 t=0 和 t=1
            t_scalar = (i + 0.5) * dt
            t_scalar = min(max(t_scalar, eps_t), 1.0 - eps_t)
            t = torch.full((batch_size,), t_scalar, device=self.device)
    
            if self.use_feature_fusion:
                model_input = torch.cat([x, cond_fused], dim=1)
            else:
                model_input = torch.cat([x, condition], dim=1)
    
            if debug:
                check_tensor("x before model", x, i, t)
                check_tensor("model_input", model_input, i, t)
    
            u_t = self.model(model_input, t)
    
            if debug:
                check_tensor("u_t", u_t, i, t)
    
            x = x + u_t * dt
    
            if debug:
                check_tensor("x after update", x, i, t)
    
        return x

    @torch.no_grad()
    def sample_heun(self, condition, shape=None, num_steps=50):
        batch_size = condition.shape[0]
        if shape is None:
            shape = (batch_size, 1, condition.shape[2], condition.shape[3])

        cond_fused = self._fuse_condition(condition)
        cond_z = condition[:, 1:2, :, :]

        if self.use_asa_fm:
            alpha = self.asa_alpha
            eps = torch.randn(shape, device=self.device)
            x = (alpha ** 0.5) * cond_z + ((1 - alpha) ** 0.5) * eps
        else:
            x = torch.randn(shape, device=self.device)

        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=self.device)

            if self.use_feature_fusion:
                inp = torch.cat([x, cond_fused], dim=1)
            else:
                inp = torch.cat([x, condition], dim=1)
            u_t = self.model(inp, t)

            x_pred = x + u_t * dt

            t_next = torch.clamp(
                torch.full((batch_size,), (i + 1) * dt, device=self.device),
                max=1.0,
            )
            if self.use_feature_fusion:
                inp_next = torch.cat([x_pred, cond_fused], dim=1)
            else:
                inp_next = torch.cat([x_pred, condition], dim=1)
            u_t_next = self.model(inp_next, t_next)

            x = x + (u_t + u_t_next) * 0.5 * dt

        return x
