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

from .noise_schedule import NoiseSchedule


def model_wrapper(
    model,
    noise_schedule: NoiseSchedule,
    time_input_type: str = "continuous",
    model_type: str = "noise",
    reverse_time: bool = False,
):
    """
    wrap a model to noise prediction function.
    """

    def noise_predict_fn(x, t_continuous):
        model_input_time = t_continuous if not reverse_time else 1.0 - t_continuous
        # get model output
        if time_input_type == "discrete_999":
            output = model(x, model_input_time * 1000.0 - 1)
        elif time_input_type == "discrete_1000":
            output = model(x, model_input_time * 1000.0)
        elif time_input_type == "continuous":
            output = model(x, model_input_time)
        else:
            raise NotImplementedError(f"time_input_type {time_input_type} not implemented")

        if len(t_continuous) != 1:
            t_continuous = t_continuous[0]
        if model_type == "noise":
            return output
        elif model_type == "data":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return (x - alpha_t * output) / sigma_t
        elif model_type == "velocity":
            if reverse_time:
                output = -output
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            d_alpha_t, d_sigma_t = noise_schedule.marginal_d_alpha(t_continuous), noise_schedule.marginal_d_std(
                t_continuous
            )
            return (d_alpha_t * x - alpha_t * output) / (d_alpha_t * sigma_t - alpha_t * d_sigma_t)
        elif model_type == "score":
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return -sigma_t * output
        else:
            raise NotImplementedError(f"model_type {model_type} not implemented")

    return noise_predict_fn
