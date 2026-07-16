# CMMD was introduced by Sadeep Jayasumana, Srikumar Ramalingam, Andreas Veit, Daniel Glasner, Ayan Chakrabarti, and Sanjiv Kumar in "Rethinking FID: Towards a Better Evaluation Metric for Image Generation", see https://arxiv.org/abs/2401.09603.
# The original implementation is by the Google Research Authors, licensed under the Apache License 2.0. See https://github.com/google-research/google-research/tree/master/cmmd.
# The PyTorch version is adapted by https://github.com/sayakpaul/cmmd-pytorch, licensed under the Apache License 2.0.

"""Embedding models used in the CMMD calculation."""

import torch

from ...utils.dist import dist_barrier, is_master

_CLIP_MODEL_NAME = "openai/clip-vit-large-patch14-336"
_CUDA_AVAILABLE = torch.cuda.is_available()


def _resize_bicubic(images, size):
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    images = torch.nn.functional.interpolate(images, size=(size, size), mode="bicubic")
    images = images.permute(0, 2, 3, 1).numpy()
    return images


class ClipEmbeddingModel:
    """CLIP image embedding calculator."""

    def __init__(self):
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        if is_master():
            self.image_processor = CLIPImageProcessor.from_pretrained(_CLIP_MODEL_NAME)
            self._model = CLIPVisionModelWithProjection.from_pretrained(_CLIP_MODEL_NAME).eval()
        dist_barrier()
        if not is_master():
            self.image_processor = CLIPImageProcessor.from_pretrained(_CLIP_MODEL_NAME)
            self._model = CLIPVisionModelWithProjection.from_pretrained(_CLIP_MODEL_NAME).eval()
        dist_barrier()
        if _CUDA_AVAILABLE:
            self._model = self._model.cuda()

        self.input_image_size = self.image_processor.crop_size["height"]

    @torch.no_grad()
    def embed(self, images):
        """Computes CLIP embeddings for the given images.

        Args:
          images: An image array of shape (batch_size, height, width, 3). Values are
            in range [0, 1].

        Returns:
          Embedding array of shape (batch_size, embedding_width).
        """
        images = _resize_bicubic(images, self.input_image_size)
        inputs = self.image_processor(
            images=images,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        )
        if _CUDA_AVAILABLE:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        image_embs = self._model(**inputs).image_embeds.cpu()
        image_embs /= torch.linalg.norm(image_embs, axis=-1, keepdims=True)
        return image_embs
