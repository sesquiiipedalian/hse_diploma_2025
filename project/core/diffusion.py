"""
Генерация изображений с использованием Stable Diffusion Img2Img.
"""
import torch
from typing import Optional
from PIL import Image


class Img2ImgProcessor:
    """
    Обёртка над StableDiffusionImg2ImgPipeline.

    Аргументы:
        model_name (str): идентификатор модели в Hugging Face.
        device (torch.device): устройство вычислений.
    """
    def __init__(self, model_name: str, device: torch.device):
        from diffusers import StableDiffusionImg2ImgPipeline
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
        )
        self.pipe = self.pipe.to(device)

    def run(self,
            init_image: Image.Image,
            prompt: str,
            strength: float = 0.75,
            guidance_scale: float = 7.5,
            num_inference_steps: int = 50
    ) -> Image.Image:
        """
        Преобразует init_image в соответствии с prompt.
        """
        result = self.pipe(
            prompt=prompt,
            init_image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        )
        return result.images[0]
