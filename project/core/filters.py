"""
Фильтрация похожих кадров по признакам.
"""
import numpy as np
from typing import List


def compute_histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Считает нормализованную гистограмму цветового изображения.
    """
    hist = []
    for ch in range(image.shape[2]):
        h, _ = np.histogram(image[:,:,ch], bins=bins, range=(0,255))
        hist.append(h)
    hist = np.concatenate(hist).astype(np.float32)
    return hist / np.sum(hist)


class SimilarityFilter:
    """
    Фильтрует похожие изображения по гистограмме.

    Аргументы:
        threshold (float): максимальное расстояние для отсева.
    """
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def filter(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Оставляет только изображения, расстояние между гистограммами
        которых выше порога.
        """
        filtered = []
        prev_hist = None
        for img in images:
            hist = compute_histogram(img)
            if prev_hist is None or np.linalg.norm(hist - prev_hist) >= self.threshold:
                filtered.append(img)
                prev_hist = hist
        return filtered
