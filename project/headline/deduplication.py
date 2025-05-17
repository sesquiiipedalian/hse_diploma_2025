"""
Удаление семантических дубликатов заголовков (GuidMaster)
"""
import logging
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from multiprocessing import Pool

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class GuidMaster:
    def __init__(self, embedder, threshold: float = 0.85,
                 method: str = 'dbscan', hierarchical_thresh: float = 0.5,
                 mmr_lambda: float = 0.7, n_jobs: int = 1):
        self.embedder = embedder
        self.threshold = threshold
        self.method = method
        self.hierarchical_thresh = hierarchical_thresh
        self.mmr_lambda = mmr_lambda
        self.n_jobs = n_jobs

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        logger.info(f"Embedding {len(texts)} texts using {self.n_jobs} workers...")
        with Pool(self.n_jobs) as pool:
            emb_list = pool.map(self.embedder.encode, texts)
        embs = torch.cat(emb_list, dim=0).cpu().numpy()
        # L2
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / (norms + 1e-8)

    def _compute_similarity(self, embs: np.ndarray) -> np.ndarray:
        logger.info("Computing cosine similarity matrix...")
        return cosine_similarity(embs)

    def _cluster_dbscan(self, sim: np.ndarray) -> np.ndarray:
        logger.info(f"Clustering with DBSCAN (eps={1-self.threshold:.3f})...")
        dmat = 1 - sim
        labels = DBSCAN(eps=1-self.threshold, min_samples=1, metric='precomputed').fit_predict(dmat)
        return labels

    def _cluster_hierarchical(self, sim: np.ndarray) -> np.ndarray:
        logger.info(f"Clustering hierarchically (threshold={self.hierarchical_thresh})...")
        dist = 1 - sim
        Z = linkage(dist, method='average')
        labels = fcluster(Z, t=1-self.hierarchical_thresh, criterion='distance') - 1
        return labels

    def _select_mmr(self, texts: list[str], sim: np.ndarray, labels: np.ndarray) -> list[str]
