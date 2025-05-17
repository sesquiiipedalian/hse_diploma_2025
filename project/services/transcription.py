"""
Транскрипция аудио с использованием Whisper.
"""
import torch
from typing import List


class WhisperTranscriber:
    """
    Обёртка над Whisper для транскрипции аудио.

    Аргументы:
        model_size (str): размер модели ('tiny', 'base', 'small', 'medium', 'large').
        device (torch.device|str): устройство для вычислений.
    """
    def __init__(self, model_size: str = 'small', device: torch.device | str = 'cpu'):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        self.device = torch.device(device)
        self.processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            f"openai/whisper-{model_size}"
        ).to(self.device)

    def transcribe(self, audio_path: str, language: str = 'en') -> List[dict]:
        """
        Транскрибирует аудио-файл.

        Аргументы:
            audio_path (str): путь к аудио-файлу (wav/mp3).
            language (str): код языка (ISO).

        Возвращает список сегментов со словарями:
        {"start": float, "end": float, "text": str}
        """
        import soundfile as sf
        audio, sr = sf.read(audio_path)
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt").to(self.device)
        # указать язык, если нужно
        generated_ids = self.model.generate(**inputs, forced_decoder_ids=self.processor.get_decoder_prompt_ids(language=language))
        # deocde
        transcript = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return [{"start": 0.0, "end": len(audio)/sr, "text": transcript}]
