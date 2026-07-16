# The original implementation is by MIT HAN Lab, licensed under the MIT License. This file is modified from https://github.com/mit-han-lab/hart/blob/main/hart/utils/safety_check.py

from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn.functional import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class PromptSafetyCheckerConfig:
    model: Optional[str] = "google/shieldgemma-2b"
    device: str = "cuda"
    threshold: float = 0.2


class PromptSafetyChecker:
    def __init__(self, cfg: PromptSafetyCheckerConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.safety_policy = """
            The prompt must not seek to generate harmful, abusive, hateful, sexually explicit, or otherwise inappropriate content targeting individuals or protected groups.
        """
        if cfg.model is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model,
                torch_dtype=torch.bfloat16,
            ).to(self.device)

    def is_dangerous(self, user_prompt: str) -> bool:
        if self.cfg.model is None:
            return False
        prompt = f"""You are a policy expert trying to help determine whether a user
        prompt is in violation of the defined safety policies.

        <start_of_turn>
        Human Question: {user_prompt.strip()}
        <end_of_turn>

        Our safety principle is defined in the below:

        {self.safety_policy.strip()}

        Does the human question violate the above principle? Your answer must start
        with 'Yes' or 'No'. And then walk through step by step to be sure we answer
        correctly.
        """

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits

        # Extract the logits for the Yes and No tokens
        vocab = self.tokenizer.get_vocab()
        selected_logits = logits[0, -1, [vocab["Yes"], vocab["No"]]]

        # Convert these logits to a probability with softmax
        probabilities = softmax(selected_logits, dim=0)

        # Return probability of 'Yes'
        score = probabilities[0].item()

        return score > self.cfg.threshold
