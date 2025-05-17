"""
services: сторонние сервисы, загрузчики и обёртки.
"""

from .downloader import ModelDownloader
from .transcription import WhisperTranscriber

__all__ = [
    "ModelDownloader",
    "WhisperTranscriber",
]
