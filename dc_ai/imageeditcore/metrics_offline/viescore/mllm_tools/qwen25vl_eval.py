# Modified from GEdit Bench introduced by Shiyu Liu et. al. See https://github.com/stepfun-ai/Step1X-Edit.
# The original implementation is licensed under the Apache License 2.0.

import random
from io import BytesIO
from typing import List

import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def process_image(image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Qwen25VL:
    def __init__(self, device) -> None:
        self.device = device
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
            torch_dtype=torch.float16,
            device_map=self.device,
            low_cpu_mem_usage=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct-AWQ")

    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        if not isinstance(image_links, list):
            image_links = [image_links]

        image_links_base64 = []

        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": img_link} for img_link in image_links]
                + [{"type": "text", "text": text_prompt}],
            }
        ]
        return messages

    def get_parsed_output(self, messages):
        set_seed(42)
        # Prepare the inputs
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        # Process inputs
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        # Generate output
        generation_config = {
            "max_new_tokens": 512,
            "num_beams": 1,
            "do_sample": False,
            "temperature": 0.1,
            "top_p": None,
        }
        generated_ids = self.model.generate(**inputs, **generation_config)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0] if output_text else ""


if __name__ == "__main__":
    model = Qwen25VL()
    prompt = model.prepare_prompt(
        ["https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"], "Describe the image in detail."
    )
    res = model.get_parsed_output(prompt)
    print("result : \n", res)
