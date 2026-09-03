# VBench was introduced by Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu in "VBench: Comprehensive Benchmark Suite for Video Generative Models", see https://arxiv.org/abs/2311.17982.
# VBench 2.0 was introduced by Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, and Ziwei Liu in "VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness", see https://arxiv.org/abs/2503.21755.
# The original implementation of VBench and VBench 2.0 is by VBench Team, licensed under the Apache License 2.0. See https://github.com/Vchitect/VBench.
# This file is modified from https://github.com/Vchitect/VBench/blob/master/vbench2_beta_i2v/utils.py, with the following changes:
# - support passing image_root


import os

from vbench2_beta_i2v.utils import load_json


def load_i2v_dimension_info(
    json_dir, dimension, lang, resolution, image_root="assets/data/vbench/vbench_i2v_imgs/crop"
):
    """
    Load video list and prompt information based on a specified dimension and language from a JSON file.

    Parameters:
    - json_dir (str): The directory path where the JSON file is located.
    - dimension (str): The dimension for evaluation to filter the video prompts.
    - lang (str): The language key used to retrieve the appropriate prompt text.
    - resulution (str): The resolution of the image will be used

    Returns:
    - video_list (list): A list of video file paths that match the specified dimension.
    - prompt_dict_ls (list): A list of dictionaries, each containing a prompt and its corresponding video list.

    The function reads the JSON file to extract video information. It filters the prompts based on the specified
    dimension and compiles a list of video paths and associated prompts in the specified language.

    Notes:
    - The JSON file is expected to contain a list of dictionaries with keys 'dimension', 'video_list', and language-based prompts.
    - The function assumes that the 'video_list' key in the JSON can either be a list or a single string value.
    """
    video_pair_list = []
    prompt_dict_ls = []
    full_prompt_list = load_json(json_dir)
    image_root = os.path.join(image_root, resolution)
    for prompt_dict in full_prompt_list:
        if dimension in prompt_dict["dimension"] and "video_list" in prompt_dict:
            prompt = prompt_dict[f"prompt_{lang}"]
            cur_video_list = (
                prompt_dict["video_list"]
                if isinstance(prompt_dict["video_list"], list)
                else [prompt_dict["video_list"]]
            )
            # create image-video pair
            if "image_name" in prompt_dict:
                image_path = os.path.join(image_root, prompt_dict["image_name"])
            elif "custom_image_path" in prompt_dict:
                image_path = prompt_dict["custom_image_path"]
            else:
                raise Exception("prompt_dict doesn't contain 'image_name' or 'custom_image_path' key")

            cur_video_pair = [(image_path, video) for video in cur_video_list]
            video_pair_list += cur_video_pair
            if "auxiliary_info" in prompt_dict and dimension in prompt_dict["auxiliary_info"]:
                prompt_dict_ls += [
                    {
                        "prompt": prompt,
                        "video_list": cur_video_list,
                        "auxiliary_info": prompt_dict["auxiliary_info"][dimension],
                    }
                ]
            else:
                prompt_dict_ls += [{"prompt": prompt, "video_list": cur_video_list}]
    return video_pair_list, prompt_dict_ls
