# CLIP Score was introduced by Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi in "CLIPScore: A Reference-free Evaluation Metric for Image Captioning", see https://arxiv.org/abs/2104.08718.
# The original implementation is by Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, Yejin Choi, licensed under the MIT License. See https://github.com/jmhessel/clipscore.

import torch
import torchvision.transforms.functional
from torch import Tensor

from ..utils.dist import is_dist_initialized, sync_tensor


class CLIPScoreStats:
    def __init__(self):
        self.score = 0.0
        self.n_samples = 0
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        import clip

        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        self.model.eval()

    def compute_batch(self, images: Tensor, text: list[str]) -> tuple[float, int]:
        """Compute CLIP score on a batch of images and text.

        Args:
            images: [N, C, H, W]
            text: a list of captions

        Returns:
            score: the sum of CLIP score
            n_samples: the number of samples
        """
        assert isinstance(images, Tensor), f"Type of images {type(images)} is not supported"
        if not isinstance(text, list) or not all(isinstance(t, str) for t in text):
            raise ValueError(f"Type of text {type(text)} is not supported")
        if images.shape[0] != len(text):
            raise ValueError(f"The number of images and captions do not match, got {images.shape[0]} and {len(text)}")

        if images.dtype == torch.uint8:
            assert images.ndim == 4, f"images.ndim {images.ndim} is not supported"
            images = torch.stack(
                [self.preprocess(torchvision.transforms.functional.to_pil_image(image)) for image in images], dim=0
            )
        else:
            assert (
                images.ndim == 4
                and images.shape[1] == 3
                and images.shape[2] == self.preprocess.transforms[0].size
                and images.shape[3] == self.preprocess.transforms[0].size
            ), f"When images.dtype is {images.dtype}, we assume images are preprocessed, but shape of images {images.shape} is invalid"

        logit_scale = self.model.logit_scale.exp()
        real_features = self.model.encode_image(images.to(self.device))
        import clip

        fake_features = self.model.encode_text(clip.tokenize(text, context_length=77, truncate=True).to(self.device))

        real_features = real_features / real_features.norm(dim=1, keepdim=True).to(torch.float32)
        fake_features = fake_features / fake_features.norm(dim=1, keepdim=True).to(torch.float32)

        score = logit_scale * (fake_features * real_features)

        return score.sum().item(), images.shape[0]

    @torch.no_grad()
    def update(self, images: Tensor, text: list[str]) -> None:
        """Update CLIP score on a batch of images and text.

        Args:
            images: [N, C, H, W]
            text: a list of captions
        """
        score, n_samples = self.compute_batch(images, text)
        self.score += score
        self.n_samples += n_samples

    def compute(self):
        num_samples, pred_score = self.n_samples, self.score
        if is_dist_initialized():
            num_samples = sync_tensor(torch.tensor(num_samples).cuda(), reduce="sum").cpu().numpy().item()
            pred_score = sync_tensor(torch.tensor(pred_score).cuda(), reduce="sum").cpu().numpy().item()
        return pred_score / num_samples
