import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import lpips as _lpips_mod
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False

try:
    from torchvision.models import inception_v3
    from scipy.linalg import sqrtm
    _FID_AVAILABLE = True
except ImportError:
    _FID_AVAILABLE = False

_lpips_net = None
_inception_model = None


def _to_01(x):
    return (x + 1.0) / 2.0


def _to_tensor(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    return x.float()


def _broadcast_mask(mask, ref):
    if mask is None:
        return None
    mask = _to_tensor(mask)
    if mask.ndim == ref.ndim - 1:
        mask = mask.unsqueeze(1)
    while mask.ndim < ref.ndim:
        mask = mask.unsqueeze(0)
    return (mask > 0.5).float().to(ref.device)


def _masked_mean(value, mask=None):
    if mask is None:
        return value.mean()
    mask = mask.expand_as(value)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def _apply_mask_image(img, mask=None):
    if mask is None:
        return img
    if img.ndim == 2:
        return img * mask.squeeze()
    return img * mask


def compute_psnr(pred, target, mask=None, data_range=1.0):
    pred = _to_01(_to_tensor(pred))
    target = _to_01(_to_tensor(target))
    mask = _broadcast_mask(mask, pred)
    mse = _masked_mean((pred - target) ** 2, mask)
    if mse.item() == 0:
        return float("inf")
    return (10.0 * torch.log10(torch.tensor(data_range ** 2, device=pred.device) / mse)).item()


def compute_mae(pred, target, mask=None):
    pred = _to_01(_to_tensor(pred))
    target = _to_01(_to_tensor(target))
    mask = _broadcast_mask(mask, pred)
    return _masked_mean(torch.abs(pred - target), mask).item()


def compute_nmse(pred, target, mask=None):
    pred = _to_01(_to_tensor(pred))
    target = _to_01(_to_tensor(target))
    mask = _broadcast_mask(mask, pred)
    num = _masked_mean((pred - target) ** 2, mask)
    den = _masked_mean(target ** 2, mask) + 1e-10
    return (num / den).item()


def compute_pcc(pred, target, mask=None):
    pred = _to_01(_to_tensor(pred))
    target = _to_01(_to_tensor(target))
    mask = _broadcast_mask(mask, pred)
    if mask is not None:
        pred = pred[mask.bool()]
        target = target[mask.bool()]
    else:
        pred = pred.reshape(-1)
        target = target.reshape(-1)
    pred_mean = pred.mean()
    target_mean = target.mean()
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    denominator = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum()) + 1e-10
    return ((pred_centered * target_centered).sum() / denominator).item()


def _gaussian_kernel_1d(size, sigma):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _create_ssim_window(window_size, channels):
    _1d = _gaussian_kernel_1d(window_size, 1.5)
    _2d = _1d.unsqueeze(1) * _1d.unsqueeze(0)
    window = _2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, window_size, window_size)
    return window.contiguous()


def compute_ssim(pred, target, mask=None, window_size=11, data_range=1.0):
    pred = _to_01(_to_tensor(pred))
    target = _to_01(_to_tensor(target))
    if pred.ndim == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)
    elif pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    mask = _broadcast_mask(mask, pred)
    pred = _apply_mask_image(pred, mask)
    target = _apply_mask_image(target, mask)
    channels = pred.shape[1]
    window = _create_ssim_window(window_size, channels).to(pred.device)
    pad = window_size // 2
    mu_pred = F.conv2d(pred, window, padding=pad, groups=channels)
    mu_target = F.conv2d(target, window, padding=pad, groups=channels)
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_cross = mu_pred * mu_target
    sigma_pred_sq = F.conv2d(pred ** 2, window, padding=pad, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target ** 2, window, padding=pad, groups=channels) - mu_target_sq
    sigma_cross = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_cross
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_cross + c1) * (2 * sigma_cross + c2)) / ((mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2))
    return ssim_map.mean().item()


def _get_lpips_net(device):
    global _lpips_net
    if _lpips_net is None:
        _lpips_net = _lpips_mod.LPIPS(net='alex', verbose=False).to(device)
        _lpips_net.eval()
    return _lpips_net


def compute_lpips(pred, target, mask=None, device=None):
    if not _LPIPS_AVAILABLE:
        return None
    pred = _to_tensor(pred)
    target = _to_tensor(target)
    if device is None:
        device = pred.device
    if pred.ndim == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)
    elif pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    mask = _broadcast_mask(mask, pred)
    pred = _apply_mask_image(pred, mask)
    target = _apply_mask_image(target, mask)
    pred_3ch = pred.expand(-1, 3, -1, -1).to(device)
    target_3ch = target.expand(-1, 3, -1, -1).to(device)
    net = _get_lpips_net(device)
    with torch.no_grad():
        return net(pred_3ch, target_3ch).mean().item()


def _get_inception_model(device):
    global _inception_model
    if _inception_model is None:
        _inception_model = inception_v3(pretrained=True, transform_input=False)
        _inception_model.fc = nn.Identity()
        _inception_model.to(device).eval()
    return _inception_model


def _extract_inception_features(images, device):
    model = _get_inception_model(device)
    if images.shape[1] == 1:
        images = images.expand(-1, 3, -1, -1)
    images = _to_01(images)
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    with torch.no_grad():
        feats = model(images.to(device))
    return feats.cpu().numpy()


def compute_fid(pred_feats, target_feats):
    if not _FID_AVAILABLE:
        return None
    mu1, sigma1 = pred_feats.mean(axis=0), np.cov(pred_feats, rowvar=False)
    mu2, sigma2 = target_feats.mean(axis=0), np.cov(target_feats, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def compute_batch_metrics(pred_batch, target_batch, mask=None, device=None):
    batch_size = pred_batch.shape[0]
    ssim_list, psnr_list, mae_list, lpips_list = [], [], [], []
    nmse_list, pcc_list = [], []
    for i in range(batch_size):
        cur_mask = None if mask is None else mask[i]
        ssim_list.append(compute_ssim(pred_batch[i], target_batch[i], cur_mask))
        psnr_list.append(compute_psnr(pred_batch[i], target_batch[i], cur_mask))
        mae_list.append(compute_mae(pred_batch[i], target_batch[i], cur_mask))
        nmse_list.append(compute_nmse(pred_batch[i], target_batch[i], cur_mask))
        pcc_list.append(compute_pcc(pred_batch[i], target_batch[i], cur_mask))
        lpips_list.append(compute_lpips(pred_batch[i], target_batch[i], cur_mask, device=device))
    result = {
        "ssim": sum(ssim_list) / len(ssim_list),
        "psnr": sum(psnr_list) / len(psnr_list),
        "mae": sum(mae_list) / len(mae_list),
        "nmse": sum(nmse_list) / len(nmse_list),
        "pcc": sum(pcc_list) / len(pcc_list),
    }
    valid_lpips = [v for v in lpips_list if v is not None]
    result["lpips"] = sum(valid_lpips) / len(valid_lpips) if valid_lpips else None
    return result
