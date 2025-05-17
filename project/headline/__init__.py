"""
Пакет headline: дообучение и дедупликация заголовков.
"""

from .train import HeadlineTrainer
from .deduplication import GuidMaster
from .utils import DatasetLoader, EmbeddingModel

__all__ = [
    "HeadlineTrainer",
    "GuidMaster",
    "DatasetLoader",
    "EmbeddingModel",
]
