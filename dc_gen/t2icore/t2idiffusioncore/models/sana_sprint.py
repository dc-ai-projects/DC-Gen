# SANA was introduced by Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, and Song Han in "SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers", see https://arxiv.org/abs/2410.10629.
# The original implementation is by NVIDIA CORPORATION & AFFILIATES, licensed under the Apache License 2.0. See https://github.com/NVlabs/Sana/blob/main/app/sana_sprint_pipeline.py.
from dataclasses import dataclass
from typing import Optional

import torch

from ....apps.utils.dist import is_master
from ....models.utils.network import get_device
from .sana_t2i import SanaT2I, SanaT2IConfig


@dataclass
class SanaSprintConfig(SanaT2IConfig):
    qk_norm: bool = True
    cross_norm: bool = True

    max_timesteps: Optional[float] = 1.57080
    intermediate_timesteps: Optional[float] = 1.3
    timesteps: Optional[list[float]] = None
    num_inference_steps: int = 2

    cfg_embed_scale: float = 0.1

    # scm scheduler specific
    eval_scheduler: str = "SCMScheduler"
    sigma_data: float = 0.5


class SanaSprint(SanaT2I):
    def __init__(self, cfg: SanaSprintConfig):
        super().__init__(cfg)
        self.cfg: SanaSprintConfig

    def build_model(self):
        from ...models.ops.timestep_embed import TimestepEmbedder

        super().build_model()
        self.cfg_embedder = TimestepEmbedder(hidden_size=self.cfg.hidden_size)

    def load_model(self):
        checkpoint = torch.load(self.cfg.pretrained_path, map_location="cpu", weights_only=False)
        if self.cfg.pretrained_source == "sana-sprint":
            state_dict = checkpoint["state_dict"]
            del state_dict["logvar_linear.weight"]
            del state_dict["logvar_linear.bias"]
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            if is_master():
                if len(missing) > 0:
                    print(f"Missing keys in the checkpoint: {missing}")
                if len(unexpected) > 0:
                    print(f"Unexpected keys in the checkpoint: {unexpected}")
        else:
            raise ValueError(f"Pretrained source {self.cfg.pretrained_source} is not supported")

    def flow_forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, cfg_scale: float
    ) -> torch.Tensor:
        assert mask is not None
        height, width = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        in_channels = x.shape[1]

        y = y.to(self.y_embedder.y_proj.fc1.weight.dtype)
        x = x.to(y.dtype)

        x = self.x_embedder(x)

        cfg_scale = cfg_scale * torch.ones_like(t)

        t = self.t_embedder(t)  # (N, D)
        cfg_embed = self.cfg_embedder(cfg_scale * self.cfg.cfg_embed_scale)  # (1, D)
        t = t + cfg_embed  # (N, D)

        t0 = self.t_block(t)

        y = self.y_embedder(y, mask=mask)  # (N, D)
        y = self.attention_y_norm(y)

        if mask.shape[0] != y.shape[0]:
            if y.shape[0] % mask.shape[0] == 0:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            else:
                raise ValueError(f"First channel of mask must be a factor of the first channel of y.")

        for block in self.blocks:
            x = block(x, y, t0, mask, (height, width))  # (N, T, D) #support grad checkpoint

        x = self.final_layer(x, t, out_channels=in_channels)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x, in_channels, height, width)  # (N, out_channels, H, W)

        return x

    def forward_without_cfg(self, x, t, y, mask):
        raise NotImplementedError("SanaSprint distills cfg scale into the model")

    def forward_with_cfg(self, x, t, y, mask, cfg_scale) -> torch.Tensor:
        # TrigFlow --> Flow Transformation
        # the input now is [0, np.pi/2], arctan(N(P_mean, P_std))
        t = torch.sin(t) / (torch.cos(t) + torch.sin(t))

        pretrain_timestep = t
        t = t.view(-1, 1, 1, 1)

        x = x * torch.sqrt(t**2 + (1 - t) ** 2)
        # forward in original flow
        model_out = self.flow_forward(x, pretrain_timestep, y, mask, cfg_scale)

        # Flow --> TrigFlow Transformation
        trigflow_model_out = ((1 - 2 * t) * x + (1 - 2 * t + 2 * t**2) * model_out) / torch.sqrt(t**2 + (1 - t) ** 2)
        return trigflow_model_out

    @torch.no_grad()
    def generate(
        self,
        text_embed_info: dict[str, dict[str, torch.Tensor]],
        noise: Optional[torch.Tensor] = None,
        cfg_scale: float = 4.5,
        pag_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        text_embeddings = text_embed_info[self.cfg.text_encoder_name]["text_embeddings"]
        text_embedding_masks = text_embed_info[self.cfg.text_encoder_name]["text_embedding_masks"]

        device = get_device(self)
        bs = text_embeddings.shape[0]

        if noise is None:
            latents = (
                torch.randn(
                    bs,
                    self.cfg.in_channels,
                    self.cfg.input_size,
                    self.cfg.input_size,
                    device=device,
                    generator=generator,
                )
                * self.cfg.sigma_data
            )
        else:
            latents = noise

        model_kwargs = dict(mask=text_embedding_masks, cfg_scale=cfg_scale)

        if self.cfg.eval_scheduler == "SCMScheduler":
            from ...scheduler.scm_scheduler import SCMScheduler

            scheduler = SCMScheduler()
            scheduler.set_timesteps(
                num_inference_steps=self.cfg.num_inference_steps,
                max_timesteps=self.cfg.max_timesteps,
                intermediate_timesteps=self.cfg.intermediate_timesteps,
                timesteps=self.cfg.timesteps,
                device=device,
            )
            timesteps = scheduler.timesteps
            sigma_data = self.cfg.sigma_data

            for i, t in list(enumerate(timesteps[:-1])):
                timestep = t.expand(bs).to(device)

                model_pred = sigma_data * self.forward_with_cfg(
                    latents / sigma_data,
                    timestep,
                    text_embeddings,
                    **model_kwargs,
                )

                latents, denoised = scheduler.step(
                    model_pred, i, t, latents, generator=generator, sigma_data=sigma_data, return_dict=False
                )

            return denoised / sigma_data
        else:
            raise ValueError(f"Eval scheduler {self.cfg.eval_scheduler} is not supported.")

    def forward(
        self,
        x: torch.Tensor,
        text_embed_info: dict[str, dict[str, torch.Tensor]],
        generator: Optional[torch.Generator] = None,
    ) -> tuple[dict[int, torch.Tensor], dict]:
        raise NotImplementedError("SanaSprint training is not supported")
