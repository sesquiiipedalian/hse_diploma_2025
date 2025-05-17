"""
Генерация и сохранение обоев на основе обработанных кадров и фильтрации.
"""

import os
from PIL import Image
from typing import List


class WallpaperProcessor:
    """
    Создание обоев.

    Аргументы:
        output_dir (str): папка для сохранения.
        size (tuple[int,int]): размер обоев.
    """
    def __init__(self, output_dir: str, size: tuple[int,int]):
        self.output_dir = output_dir
        self.size = size
        os.makedirs(output_dir, exist_ok=True)

    def process(self, images: List[Image.Image], prefix: str = 'wallpaper') -> List[str]:
        """
        Обрабатывает список PIL-изображений: ресайзит и сохраняет.

        Возвращает список путей до файлов.
        """
        paths: List[str] = []
        for idx, img in enumerate(images):
            img_resized = img.resize(self.size, Image.LANCZOS)
            filename = f"{prefix}_{idx:03d}.png"
            path = os.path.join(self.output_dir, filename)
            img_resized.save(path)
            paths.append(path)
        return paths
