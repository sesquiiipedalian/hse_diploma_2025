"""
Детекция объектов с помощью модели YOLO.
"""

import torch
import numpy as np
from typing import List, Dict, Any


class YOLODetector:
    """
    Обёртка над моделью YOLO (Ultralytics).

    Аргументы:
        weights (str): путь к весам или идентификатор версии (например, 'yolov5s').
        device (torch.device|str): устройство для вычислений.
        conf_threshold (float): порог уверенности детекции.
    """
    def __init__(self,
                 weights: str = 'yolov5s',
                 device: torch.device | str = 'cpu',
                 conf_threshold: float = 0.25):
                   
        # Загрузка
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=weights)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.conf = conf_threshold

  
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Детекция на одном изображении.

        Возвращает список словарей с ключами:
        xmin, ymin, xmax, ymax, confidence, class, name
        """
        results = self.model(image)
        df = results.pandas().xyxy[0]
        return df.to_dict(orient='records')

  
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """
        Пакетная детекция изображений.

        Возвращает список списков детекций для каждого изображения.
        """
        results = self.model(images)
        output: List[List[Dict[str, Any]]] = []
        for df in results.pandas().xyxy:
            output.append(df.to_dict(orient='records'))
        return output
