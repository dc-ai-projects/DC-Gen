# DPM-Solver was introduced by Lu, Cheng and Zhou, Yuhao and Bao, Fan and Chen, Jianfei and Li, Chongxuan and Zhu, Jun in "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps", see https://arxiv.org/abs/2206.00927.
# The original implementation is by Cheng Lu, licensed under the MIT License. See https://github.com/LuChengTHU/dpm-solver.

import math

import torch


class NoiseSchedule:
    def marginal_log_mean_coeff(self, t):
        raise NotImplementedError

    def marginal_alpha(self, t):
        raise NotImplementedError

    def marginal_std(self, t):
        raise NotImplementedError

    def marginal_lambda(self, t):
        raise NotImplementedError

    def inverse_lambda(self, lamb):
        raise NotImplementedError

    def marginal_d_alpha(self, t):
        raise NotImplementedError

    def marginal_d_std(self, t):
        raise NotImplementedError


class NoiseScheduleVP(NoiseSchedule):
    def __init__(
        self,
        schedule="discrete",
        betas=None,
        alphas_cumprod=None,
        continuous_beta_0=0.1,
        continuous_beta_1=20.0,
        dtype=torch.float32,
    ):
        """Create a wrapper class for the forward SDE (VP type).

        ***
        Update: We support discrete-time diffusion models by implementing a picewise linear interpolation for log_alpha_t.
                We recommend to use schedule='discrete' for the discrete-time diffusion models, especially for high-resolution images.
        ***

        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:

            log_alpha_t = self.marginal_log_mean_coeff(t)
            sigma_t = self.marginal_std(t)
            lambda_t = self.marginal_lambda(t)

        Moreover, as lambda(t) is an invertible function, we also support its inverse function:

            t = self.inverse_lambda(lambda_t)

        ===============================================================

        We support both discrete-time DPMs (trained on n = 0, 1, ..., N-1) and continuous-time DPMs (trained on t in [t_0, T]).

        1. For discrete-time DPMs:

            For discrete-time DPMs trained on n = 0, 1, ..., N-1, we convert the discrete steps to continuous time steps by:
                t_i = (i + 1) / N
            e.g. for N = 1000, we have t_0 = 1e-3 and T = t_{N-1} = 1.
            We solve the corresponding diffusion ODE from time T = 1 to time t_0 = 1e-3.

            Args:
                betas: A `torch.Tensor`. The beta array for the discrete-time DPM. (See the original DDPM paper for details)
                alphas_cumprod: A `torch.Tensor`. The cumprod alphas for the discrete-time DPM. (See the original DDPM paper for details)

            Note that we always have alphas_cumprod = cumprod(1 - betas). Therefore, we only need to set one of `betas` and `alphas_cumprod`.

            **Important**:  Please pay special attention for the args for `alphas_cumprod`:
                The `alphas_cumprod` is the \hat{alpha_n} arrays in the notations of DDPM. Specifically, DDPMs assume that
                    q_{t_n | 0}(x_{t_n} | x_0) = N ( \sqrt{\hat{alpha_n}} * x_0, (1 - \hat{alpha_n}) * I ).
                Therefore, the notation \hat{alpha_n} is different from the notation alpha_t in DPM-Solver. In fact, we have
                    alpha_{t_n} = \sqrt{\hat{alpha_n}},
                and
                    log(alpha_{t_n}) = 0.5 * log(\hat{alpha_n}).


        2. For continuous-time DPMs:

            We support the linear VPSDE for the continuous time setting. The hyperparameters for the noise
            schedule are the default settings in Yang Song's ScoreSDE:

            Args:
                beta_min: A `float` number. The smallest beta for the linear schedule.
                beta_max: A `float` number. The largest beta for the linear schedule.
                T: A `float` number. The ending time of the forward process.

        ===============================================================

        Args:
            schedule: A `str`. The noise schedule of the forward SDE. 'discrete' for discrete-time DPMs,
                    'linear' for continuous-time DPMs.
        Returns:
            A wrapper object of the forward SDE (VP type).

        ===============================================================

        Example:

        # For discrete-time DPMs, given betas (the beta array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', betas=betas)

        # For discrete-time DPMs, given alphas_cumprod (the \hat{alpha_n} array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', alphas_cumprod=alphas_cumprod)

        # For continuous-time DPMs (VPSDE), linear schedule:
        >>> ns = NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20.)

        """

        if schedule not in ["discrete", "linear", "cosine"]:
            raise ValueError(
                "Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear'".format(schedule)
            )

        self.schedule = schedule
        if schedule == "discrete":
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
                device = log_alphas.device
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
                device = log_alphas.device
            self.T = 1.0
            self.log_alpha_array = (
                self.numerical_clip_alpha(log_alphas)
                .reshape(
                    (
                        1,
                        -1,
                    )
                )
                .to(dtype=dtype)
            )
            self.total_N = self.log_alpha_array.shape[1]
            self.t_array = torch.linspace(0.0, 1.0, self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype).to(device)
            self.t_flat = self.t_array.flatten()
            self.log_alpha_flat = self.log_alpha_array.flatten()
            # compute the derivative of log_alpha w.r.t. t
            d_log_alpha = self.log_alpha_flat[1:] - self.log_alpha_flat[:-1]
            dt = self.t_flat[1:] - self.t_flat[:-1]
            self.d_log_alpha_dt_flat = d_log_alpha / dt
        elif schedule == "linear":
            self.T = 1.0
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
        elif schedule == "cosine":
            self.T = 0.9946
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.0
            self.cosine_t_max = (
                math.atan(self.cosine_beta_max * (1.0 + self.cosine_s) / math.pi)
                * 2.0
                * (1.0 + self.cosine_s)
                / math.pi
                - self.cosine_s
            )
            self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1.0 + self.cosine_s) * math.pi / 2.0))

    def numerical_clip_alpha(self, log_alphas, clipped_lambda=-5.1):
        """
        For some beta schedules such as cosine schedule, the log-SNR has numerical isssues.
        We clip the log-SNR near t=T within -5.1 to ensure the stability.
        Such a trick is very useful for diffusion models with the cosine schedule, such as i-DDPM, guided-diffusion and GLIDE.
        """
        log_sigmas = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_alphas))
        lambs = log_alphas - log_sigmas
        idx = torch.searchsorted(torch.flip(lambs, [0]), clipped_lambda)
        if idx > 0:
            log_alphas = log_alphas[:-idx]
        return log_alphas

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == "discrete":
            return interpolate_fn(
                t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)
            ).reshape((-1))
        elif self.schedule == "linear":
            return -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == "cosine":
            log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1.0 + self.cosine_s) * math.pi / 2.0))
            log_alpha_t = log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1.0 - torch.exp(2.0 * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == "linear":
            tmp = 2.0 * (self.beta_1 - self.beta_0) * torch.logaddexp(-2.0 * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0**2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == "discrete":
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2.0 * lamb)
            t = interpolate_fn(
                log_alpha.reshape((-1, 1)),
                torch.flip(self.log_alpha_array.to(lamb.device), [1]),
                torch.flip(self.t_array.to(lamb.device), [1]),
            )
            return t.reshape((-1,))
        elif self.schedule == "cosine":
            log_alpha = -0.5 * torch.logaddexp(-2.0 * lamb, torch.zeros((1,)).to(lamb))
            t_fn = (
                lambda log_alpha_t: torch.arccos(torch.exp(log_alpha_t + self.cosine_log_alpha_0))
                * 2.0
                * (1.0 + self.cosine_s)
                / math.pi
                - self.cosine_s
            )
            t = t_fn(log_alpha)
            return t

    def marginal_d_alpha(self, t):
        if self.schedule == "discrete":
            t_flat = self.t_flat.to(t.device)
            d_log_alpha_dt_flat = self.d_log_alpha_dt_flat.to(t.device)
            indices = torch.searchsorted(t_flat, t, side="right") - 1
            indices = torch.clamp(indices, 0, len(d_log_alpha_dt_flat) - 1)
            d_log_alpha_dt = d_log_alpha_dt_flat[indices]
            alpha = self.marginal_alpha(t)
            return alpha * d_log_alpha_dt
        elif self.schedule == "linear":
            log_alpha = self.marginal_log_mean_coeff(t)
            alpha = torch.exp(log_alpha)
            d_log_alpha_dt = -0.5 * self.beta_0 - 0.5 * (self.beta_1 - self.beta_0) * t
            return alpha * d_log_alpha_dt
        elif self.schedule == "cosine":
            u = (t + self.cosine_s) / (1.0 + self.cosine_s) * (math.pi / 2)
            tan_u = torch.tan(u)
            d_log_alpha_dt = -tan_u * (math.pi / (2 * (1.0 + self.cosine_s)))
            alpha = self.marginal_alpha(t)
            return alpha * d_log_alpha_dt
        else:
            raise NotImplementedError(f"Unsupported schedule {self.schedule}")

    def marginal_d_std(self, t):
        alpha = self.marginal_alpha(t)
        sigma = self.marginal_std(t)
        d_alpha_dt = self.marginal_d_alpha(t)
        epsilon = torch.tensor(1e-8, device=sigma.device)
        sigma_safe = torch.maximum(sigma, epsilon)
        return (-alpha * d_alpha_dt) / sigma_safe


class NoiseScheduleLinear(NoiseSchedule):
    """
    Assume t in [0, 1].
    """

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        return torch.log(1 - t)

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return 1 - t

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return t

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        return torch.log(1 - t) - torch.log(t)

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        return 1 / (torch.exp(lamb) + 1)

    def marginal_d_alpha(self, t):
        return -torch.ones_like(t)

    def marginal_d_std(self, t):
        return torch.ones_like(t)


class NoiseScheduleCosine(NoiseSchedule):
    """
    Assume t in [0, 1]. alpha_t = cos(pi/2 * t), sigma_t = sin(pi/2 * t)
    """

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        return torch.log(torch.cos(math.pi / 2 * t))

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.cos(math.pi / 2 * t)

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sin(math.pi / 2 * t)

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        Here, lambda_t = log(alpha_t/sigma_t) = -log(tan(pi/2*t))
        """
        return -torch.log(torch.tan(math.pi / 2 * t))

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        return 2 / math.pi * torch.atan(torch.exp(-lamb))

    def marginal_d_alpha(self, t):
        return -math.pi / 2 * torch.sin(math.pi / 2 * t)

    def marginal_d_std(self, t):
        return math.pi / 2 * torch.cos(math.pi / 2 * t)


def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K),
            torch.tensor(K - 2, device=x.device),
            cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K),
            torch.tensor(K - 2, device=x.device),
            cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand
