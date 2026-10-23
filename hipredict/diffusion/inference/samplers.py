import torch

from hipredict.diffusion.config import InferenceConfig
from hipredict.diffusion.models.utils import DDPMScheduler, fm_euler_step


def _inference_ddpm(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    x_source: torch.Tensor,
    edge: torch.Tensor,
    device: torch.device,
    cfg_inference: InferenceConfig,
):
    """
    Run full DDPM reverse sampling.
    """

    model.eval()

    label = 0 if cfg_inference.direction == "ir2red" else 1
    B = x_source.size(0)
    direction = torch.full((B,), label, device=device, dtype=torch.long)

    # Noise
    x = torch.randn_like(x_source)

    for t in reversed(range(scheduler.num_timesteps)):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

        with torch.no_grad():
            eps_pred = model(x, x_source, edge, direction, t_tensor)

        # Compute the mean and variance for the reverse diffusion step
        alpha_bar_t = scheduler.alpha_bar[t]
        alpha_bar_prev_t = scheduler.alpha_bar_prev[t]
        beta_t = scheduler.betas[t]

        mean = (1 / torch.sqrt(scheduler.alphas[t])) * (
            x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_pred
        )
        if t > 0:
            noise = torch.randn_like(x)
            var = beta_t * (1 - alpha_bar_prev_t) / (1 - alpha_bar_t)
            x = mean + torch.sqrt(var) * noise
        else:
            x = mean

    return x


def _inference_fm(
    model: torch.nn.Module,
    x_source: torch.Tensor,
    edge: torch.Tensor,
    mode: str,
    device: torch.device,
    cfg_inference: InferenceConfig,
    mu_anchor: torch.Tensor = None,
):
    """
    Integrate the FM ODE from t=0 to t=1.

    If ``mu_anchor`` is given, the mean of the clean estimate is moved to it
    after every step.
    """
    model.eval()

    label = 0 if cfg_inference.direction == "ir2red" else 1
    B = x_source.size(0)
    direction = torch.full((B,), label, device=device, dtype=torch.long)

    # Starting point for ODE integration
    if mode == "Condition":
        # In Condition mode, the ODE starts from noise
        x = torch.randn_like(x_source)
    else:
        # In Bridge mode, the ODE starts from the source image
        x = x_source.clone()

    dt = 1.0 / cfg_inference.FM_ODE_steps

    @torch.no_grad()
    def velocity_fn(x_in, t_scalar):
        t_tensor = torch.full((B,), t_scalar, device=device, dtype=torch.float32)
        return model(x_in, x_source, edge, direction, t_tensor)

    solver = cfg_inference.FM_solver

    for step in range(cfg_inference.FM_ODE_steps):
        t = step / cfg_inference.FM_ODE_steps

        x_prev = x
        if solver == "euler":
            velocity = velocity_fn(x, t)
            v_first = velocity
            x = fm_euler_step(velocity, x, dt)
        elif solver == "rk4":
            k1 = velocity_fn(x, t)
            v_first = k1
            k2 = velocity_fn(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = velocity_fn(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = velocity_fn(x + dt * k3, t + dt)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            raise ValueError(f"Unknown FM solver: {solver!r}. Use 'euler' or 'rk4'.")

        # Anchor the clean estimate x1_pred to the dc_correct mu
        if mu_anchor is not None:
            t_next = (step + 1) * dt
            x1_pred = x_prev + (1.0 - t) * v_first
            delta = mu_anchor - x1_pred.mean(dim=(1, 2, 3), keepdim=True)
            x = x + t_next * delta

    return x
