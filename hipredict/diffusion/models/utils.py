import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import canny


class DDPMScheduler(nn.Module):
    """
    DDPM noise scheduler.

    Compute noise:
        x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 1e-2,
    ):
        super().__init__()

        self.num_timesteps = num_timesteps
        self.register_buffer(
            "betas", torch.linspace(beta_start, beta_end, num_timesteps)
        )
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alpha_bar", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(self.alpha_bar))
        self.register_buffer(
            "sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - self.alpha_bar)
        )

        self.register_buffer(
            "alpha_bar_prev", torch.cat([torch.tensor([1.0]), self.alpha_bar[:-1]])
        )
        self.register_buffer(
            "posterior_variance",
            self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar),
        )

    def add_noise(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise to the input image x0 at time step t using noise.
        """
        alpha_bar = self.alpha_bar[t][:, None, None, None]
        noisy_x = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
        return noisy_x


def fm_interpolate(
    x_1: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Linear interpolation path (rectified flow):
    """
    t = t[:, None, None, None]
    return (1.0 - t) * noise + t * x_1


def fm_velocity_target(
    x_1: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """
    Target velocity for the linear interpolation path:
    """
    return x_1 - noise


def fm_euler_step(
    v_pred: torch.Tensor,
    x_t: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """
    Euler ODE integration step
    """
    return x_t + v_pred * dt


def sobel_gradients(x: torch.Tensor, padding_mode: str = "zeros"):
    """
    Signed Sobel gradients (gx, gy).
    """
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)

    if padding_mode == "zeros":
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
    else:
        xp = F.pad(x, (1, 1, 1, 1), mode=padding_mode)
        gx = F.conv2d(xp, kx)
        gy = F.conv2d(xp, ky)
    return gx, gy


def sobel_magnitude(x: torch.Tensor, padding_mode: str = "zeros") -> torch.Tensor:
    """
    Raw (un-normalised) Sobel gradient magnitude, used for edge conditioning.
    """
    gx, gy = sobel_gradients(x, padding_mode)
    return torch.sqrt(gx**2 + gy**2 + 1e-8)


def compute_edge_sobel(x: torch.Tensor):
    """
    Compute edge from source image using Sobel
    """
    mag = sobel_magnitude(x)

    # Normalise per image to [0, 1]
    B = mag.shape[0]
    mn = mag.view(B, -1).min(dim=1).values[:, None, None, None]
    mx = mag.view(B, -1).max(dim=1).values[:, None, None, None]
    return (mag - mn) / (mx - mn + 1e-8)


def compute_edge_dexined(x: torch.Tensor, model: nn.Module):
    """
    Compute edge from source image using DexiNed model
    """
    model.eval()

    with torch.no_grad():
        # Normalise input per-image to [0, 1] for DexiNed
        B = x.shape[0]
        flat = x.view(B, -1)
        mn = flat.min(dim=1).values[:, None, None, None]
        mx = flat.max(dim=1).values[:, None, None, None]
        x = (x - mn) / (mx - mn + 1e-8)

        # Duplicate to 3 channels
        x = x.repeat(1, 3, 1, 1)

        # kornia's DexiNed returns the fused edge map directly as [B, 1, H, W]
        # (not a list of side-outputs), so do not index with [-1].
        edge = torch.sigmoid(model(x))  # [B, 1, H, W]

        # Per-image normalisation to [0, 1]
        flat_e = edge.view(B, -1)
        mn_e = flat_e.min(dim=1).values[:, None, None, None]
        mx_e = flat_e.max(dim=1).values[:, None, None, None]
        return (edge - mn_e) / (mx_e - mn_e + 1e-8)


def compute_edge_soft_canny(
    x: torch.Tensor,
    sigma_value: float = 1.8,
    low: float = 0.20,
    high: float = 0.40,
    qlo: float = 0.01,
    qhi: float = 0.99,
):
    """
    Compute edge from source image using a soft Canny edge detector.

    The source is first robustly rescaled to [0, 1] using the 1-99 percentiles normalization,
    so the Canny thresholds are meaningful regardless of the input
    domain. The gradient magnitude is then gated by the hysteresis mask, thin
    border artifacts are removed, and the result is per-image normalised to
    [0, 1] via a robust upper quantile.
    """
    # Robust per-image [0, 1] rescale (percentile-based, outlier tolerant).
    flat = x.flatten(1)
    lo = torch.quantile(flat, qlo, dim=1)[:, None, None, None]
    hi = torch.quantile(flat, qhi, dim=1)[:, None, None, None]
    x = ((x - lo) / (hi - lo + 1e-8)).clamp(0, 1)

    # Soft Canny: gradient magnitude gated by the hysteresis (binary) mask.
    sigma = torch.tensor([[sigma_value, sigma_value]], device=x.device, dtype=x.dtype)
    mag, binary = canny(x, sigma=sigma, low_threshold=low, high_threshold=high)
    soft = mag * binary

    # Remove thin border artifacts left by the convolutions.
    soft = soft.clone()
    soft[..., :2, :] = 0
    soft[..., -2:, :] = 0
    soft[..., :, :2] = 0
    soft[..., :, -2:] = 0

    # Per-image normalisation to [0, 1] via a robust upper quantile.
    hi_q = torch.quantile(soft.flatten(1), 0.995, dim=1)[:, None, None, None]
    return (soft / (hi_q + 1e-8)).clamp(0, 1)


def load_edge_model(edge_mode: str, device) -> nn.Module:
    """Return the pretrained DexiNed model for edge_mode='dexined', else None."""
    if edge_mode != "dexined":
        return None
    from kornia.models.dexined import DexiNed

    return DexiNed(pretrained=True).to(device).eval()


def compute_edge(x: torch.Tensor, edge_mode: str, edge_model: nn.Module = None):
    """Edge conditioning channel for the source image, or None for edge_mode='none'."""
    with torch.no_grad():
        if edge_mode == "sobel":
            return compute_edge_sobel(x)
        if edge_mode == "dexined":
            if edge_model is None:
                raise ValueError("edge_mode='dexined' needs the DexiNed model")
            return compute_edge_dexined(x, edge_model)
        if edge_mode == "soft_canny":
            return compute_edge_soft_canny(x)
        if edge_mode == "none":
            return None
    raise ValueError(f"Unknown edge mode: {edge_mode}")
