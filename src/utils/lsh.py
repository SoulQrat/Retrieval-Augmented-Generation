import numpy as np
from typing import List
from src.utils import cosine_similarity

class LSH:
    def __init__(self, L: int, k: int, embed_dim: int):
        self.L = L
        self.k = k
        self.embed_dim = embed_dim
        self.hash_functions = np.random.rand(self.L, self.k, embed_dim)
        self.data = [{} for _ in range(self.L)]

    def _get_hash(self, obj: np.ndarray, hash_function: np.ndarray) -> tuple:
        hash = [0 for _ in range(self.k)]
        for i, v in enumerate(hash_function):
            hash[i] = 1 if cosine_similarity(v, obj) >= 0 else 0
        return tuple(hash)

    def add(self, obj: np.ndarray) -> None:
        for i, hash_function in enumerate(self.hash_functions):
            hash = self._get_hash(obj, hash_function)
            if hash not in self.data[i]:
                self.data[i][hash] = []
            self.data[i][hash].append(obj)

    def find(self, obj: np.ndarray, n: int = 1, get_dist: bool=False) -> List[np.ndarray]:
        neighbours = []
        for i, hash_function in enumerate(self.hash_functions):
            hash = self._get_hash(obj, hash_function)
            neighbours.extend(self.data[i].get(hash, []))
        dist = [-cosine_similarity(neighbour, obj) for neighbour in neighbours]
        idx = np.argsort(dist)[:n]
        if get_dist:
            return [neighbours[i] for i in idx], [-dist[i] for i in idx]
        return [neighbours[i] for i in idx]
