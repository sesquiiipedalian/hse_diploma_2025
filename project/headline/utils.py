"""
Утилиты: загрузка данных и модель эмбеддингов.
"""
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch


def load_lenta_dataset(path: str) -> pd.DataFrame:
    """Загрузить датасет Lenta.ru с колонками ['text','headline']"""
    df = pd.read_csv(path)
    return df[['text', 'headline']]


class EmbeddingModel:
    """
    Комбинированный энкодер: BERT, RoBERTa, E5.
    """
    def __init__(self, names: list[str], device: torch.device):
        self.models = {}
        self.tokenizers = {}
        self.device = device
        for name in names:
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModel.from_pretrained(name).to(device)
            self.tokenizers[name] = tok
            self.models[name] = mdl.eval()

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        embs = []
        for name, tok in self.tokenizers.items():
            inp = tok(text, return_tensors='pt', truncation=True, padding=True).to(self.device)
            out = self.models[name](**inp, output_hidden_states=False)
            emb = out.last_hidden_state.mean(dim=1)
            embs.append(emb)
        return torch.stack(embs).mean(dim=0)
