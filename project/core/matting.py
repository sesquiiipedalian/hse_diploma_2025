"""
Матирование изображения с помощью модели ModNet.
"""
import torch
from PIL import Image
import numpy as np


class ModNetMatting:
    """
    Обёртка над моделью ModNet.

    Аргументы:
        weights_path (str): путь к весам модели.
        device (torch.device): устройство для вычислений.
    """
    def __init__(self, weights_path: str, device: torch.device):
        from modnet import ModNet  # локальный импорт, чтобы не грузить весь модуль сразу
        self.device = device
        self.model = ModNet(backbone_pretrained=False)
        checkpoint = torch.load(weights_path, map_location=device)
        self.model.load_state_dict(checkpoint)
        self.model.to(device)
        self.model.eval()

    def apply(self, image: Image.Image) -> Image.Image:
        """
        Применяет матирование к изображению.

        Возвращает RGBA-изображение (фон прозрачный).
        """
        #  PIL -> numpy
        np_img = np.array(image.convert("RGB"))
        h, w = np_img.shape[:2]
        # нормализация и подготовка тензора
        inp = torch.from_numpy(np_img / 255.0).permute(2,0,1).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            matte = self.model(inp)[0][0,:,:].cpu().numpy()
        # RGBA
        alpha = (matte * 255).astype(np.uint8)
        rgba = np.dstack((np_img, alpha))
        return Image.fromarray(rgba, mode='RGBA')
