import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_fusion import FeatureAdaptiveFusion


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


class SAFECFMTrainer:
    def __init__(
        self,
        model,
        sigma_min=1e-4,
        device="cuda",
        use_feature_fusion=True,
        use_asa_fm=True,
        use_boundary_reg=True,
        asa_alpha=0.5,
        boundary_lambda=0.1,
        fusion_feat_channels=32,
    ):
        self.model = model
        self.sigma_min = sigma_min
        self.device = device

        self.use_feature_fusion = use_feature_fusion
        self.use_asa_fm = use_asa_fm
        self.use_boundary_reg = use_boundary_reg
        self.asa_alpha = asa_alpha
        self.boundary_lambda = boundary_lambda

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

    def compute_loss(self, x_1, condition):
        batch_size = x_1.shape[0]

        u = torch.rand(batch_size, device=self.device).clamp(1e-5, 1 - 1e-5)
        t = torch.sigmoid(torch.log(u / (1.0 - u)))
        t_expanded = t[:, None, None, None]

        cond_fused = self._fuse_condition(condition)
        cond_z = condition[:, 1:2, :, :]

        x_0 = self._build_base_sample(x_1, cond_z)

        x_t = t_expanded * x_1 + (1.0 - t_expanded) * x_0
        u_t = x_1 - x_0

        if self.use_feature_fusion:
            model_input = torch.cat([x_t, cond_fused], dim=1)
        else:
            model_input = torch.cat([x_t, condition], dim=1)

        predicted_u_t = self.model(model_input, t)

        mse_loss = F.mse_loss(predicted_u_t, u_t, reduction='sum')

        if self.use_boundary_reg and self.sobel is not None:
            pred_gx, pred_gy = self.sobel(predicted_u_t)
            gt_gx, gt_gy = self.sobel(u_t)
            grad_loss = F.mse_loss(pred_gx, gt_gx, reduction='sum') + \
                        F.mse_loss(pred_gy, gt_gy, reduction='sum')
            total_loss = mse_loss + self.boundary_lambda * grad_loss
        else:
            total_loss = mse_loss

        return total_loss

    @torch.no_grad()
    def sample(self, condition, shape=None, num_steps=50):
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
                model_input = torch.cat([x, cond_fused], dim=1)
            else:
                model_input = torch.cat([x, condition], dim=1)

            u_t = self.model(model_input, t)
            x = x + u_t * dt

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
