"""
Извлечение ключевых кадров из видео на основе порога изменения.
"""

import cv2
import numpy as np
from typing import List


class KeyFrameExtractor:
    """
    Экстрактор ключевых кадров.

    Аргументы:
        threshold (float): относительный порог изменения (0-1).
        min_interval (int): минимальное число кадров между ключевыми.
    """
    def __init__(self, threshold: float = 0.5, min_interval: int = 30):
        self.threshold = threshold
        self.min_interval = min_interval

    def extract(self, video_path: str) -> List[np.ndarray]:
        """
        Извлекает ключевые кадры из видео.

        Возвращает список BGR-кадров.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Не удалось открыть видео: {video_path}")

        ret, prev = cap.read()
        if not ret:
            cap.release()
            return []

        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        keyframes = [prev]
        idx = 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if idx % self.min_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(gray, prev_gray)
                score = float(np.mean(diff) / 255.0)
                if score >= self.threshold:
                    keyframes.append(frame)
                    prev_gray = gray
            idx += 1

        cap.release()
        return keyframes
