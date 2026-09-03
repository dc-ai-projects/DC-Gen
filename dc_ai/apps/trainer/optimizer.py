# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.optim as optim


class CAMEWrapper(optim.Optimizer):
    """Implements CAME algorithm.
    This implementation is based on:
    `CAME: Confidence-guided Adaptive Memory Efficient Optimization`
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): external learning rate (default: None)
        eps (tuple[float, float]): regularization constants for square gradient
            and instability respectively (default: (1e-30, 1e-16))
        clip_threshold (float): threshold of root-mean-square of
            final gradient update (default: 1.0)
        betas (tuple[float, float, float]): coefficient used for computing running averages of
        update, square gradient and instability (default: (0.9, 0.999, 0.9999)))
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-16),
        clip_threshold=1.0,
        betas=(0.9, 0.999, 0.9999),
        weight_decay=0.0,
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            betas=betas,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    def _get_options(self, param_shape):
        if len(param_shape) == 4:  # Convolutional layer
            if param_shape[2] == 1 and param_shape[3] == 1:  # 1x1 conv
                return True, "1x1_conv"
            else:  # 3x3 conv or others
                return False, "conv"
        elif len(param_shape) == 2:  # Linear layer, exactly 2D
            return True, "linear"
        return False, "other"

    def _rms(self, tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("CAME does not support sparse gradients.")

                state = self.state[p]

                grad_shape = grad.shape

                # factored = self._get_options(grad_shape)
                factored, layer_type = self._get_options(grad_shape)
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0

                    state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        if layer_type == "1x1_conv" or layer_type == "linear":
                            # 1x1 conv and linear layers can be handled in the same way
                            state["exp_avg_sq_row"] = torch.zeros(grad_shape[0]).type_as(grad)
                            state["exp_avg_sq_col"] = torch.zeros(grad_shape[1]).type_as(grad)
                            state["exp_avg_res_row"] = torch.zeros(grad_shape[0]).type_as(grad)
                            state["exp_avg_res_col"] = torch.zeros(grad_shape[1]).type_as(grad)
                        else:
                            state["exp_avg_sq"] = torch.zeros_like(grad)

                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["RMS"] = 0

                state["step"] += 1
                state["RMS"] = self._rms(p.data)

                update = (grad**2) + group["eps"][0]

                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    if layer_type == "1x1_conv" or layer_type == "linear":
                        # Handle dimensions
                        if len(grad_shape) == 4:  # 1x1 conv
                            update_reshaped = update.squeeze(-1).squeeze(-1)  # Remove the last two dimensions
                        else:
                            update_reshaped = update

                        exp_avg_sq_row.mul_(group["betas"][1]).add_(
                            update_reshaped.mean(dim=1), alpha=1.0 - group["betas"][1]
                        )
                        exp_avg_sq_col.mul_(group["betas"][1]).add_(
                            update_reshaped.mean(dim=0), alpha=1.0 - group["betas"][1]
                        )

                    # Approximate calculation
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    if layer_type == "1x1_conv":
                        # Need to reshape back to 4D
                        update = update.view(grad_shape[0], grad_shape[1], 1, 1)
                    update.mul_(grad)
                else:
                    # 3x3 conv or other cases: use standard AdamW method
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(group["betas"][1]).add_(update, alpha=1.0 - group["betas"][1])
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

                exp_avg = state["exp_avg"]
                exp_avg.mul_(group["betas"][0]).add_(update, alpha=1 - group["betas"][0])

                # Confidence-guided strategy
                # Calculation of instability
                res = (update - exp_avg) ** 2 + group["eps"][1]

                if factored:
                    exp_avg_res_row = state["exp_avg_res_row"]
                    exp_avg_res_col = state["exp_avg_res_col"]

                    if layer_type == "1x1_conv" or layer_type == "linear":
                        # Handle dimensions
                        if len(grad_shape) == 4:  # 1x1 conv
                            res_reshaped = res.squeeze(-1).squeeze(-1)  # Remove last two dimensions
                        else:
                            res_reshaped = res

                        # Update residual statistics
                        exp_avg_res_row.mul_(group["betas"][2]).add_(
                            res_reshaped.mean(dim=1), alpha=1.0 - group["betas"][2]
                        )
                        exp_avg_res_col.mul_(group["betas"][2]).add_(
                            res_reshaped.mean(dim=0), alpha=1.0 - group["betas"][2]
                        )

                    # Approximate calculation
                    res_approx = self._approx_sq_grad(exp_avg_res_row, exp_avg_res_col)
                    if layer_type == "1x1_conv":
                        # Need to reshape back to 4D
                        res_approx = res_approx.view(grad_shape[0], grad_shape[1], 1, 1)
                    update = res_approx.mul_(exp_avg)
                else:
                    update = exp_avg.clone()

                if group["weight_decay"] != 0:
                    p.data.add_(p.data, alpha=-group["weight_decay"] * group["lr"])

                update.mul_(group["lr"])
                p.data.add_(-update)

        return loss
