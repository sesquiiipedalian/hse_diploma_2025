import torch


class Device:
    """
    Класс для определения и работы с устройством вычислений.

    Аргументы:
        prefer_gpu (bool): разрешить использование GPU.
    """
  
    def __init__(self, prefer_gpu: bool = True):
        self.prefer_gpu = prefer_gpu
        self.device = self._select_device()

  
    def _select_device(self) -> torch.device:
        """
        Выбирает лучшее доступное устройство.
        """
        if self.prefer_gpu and torch.cuda.is_available():
            return torch.device("cuda")
        try:
            # Apple Silicon
            if self.prefer_gpu and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
        except Exception:
            pass
        return torch.device("cpu")

  
    def __str__(self):
        return f"Device({self.device.type})"

  
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

  
    def is_mps(self) -> bool:
        return self.device.type == "mps"

  
    def is_cpu(self) -> bool:
        return self.device.type == "cpu"

  
    def get(self) -> torch.device:
        """
        Возвращает объект torch.device.
        """
        return self.device


def get_default_device(prefer_gpu: bool = True) -> torch.device:
    """
    Удобная функция для быстрого получения устройства.
    """
    return Device(prefer_gpu).get()
