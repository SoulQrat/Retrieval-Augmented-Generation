import numpy as np
from typing import List
from src.utils import cosine_similarity

class LS:
    def __init__(self):
        self.data = []

    def add(self, obj: np.ndarray) -> None:
        self.data.append(np.asarray(obj))

    def find(self, obj: np.ndarray, n: int=1) -> List[np.ndarray]:
        dist = [-cosine_similarity(neighbour, obj) for neighbour in self.data]
        dist = np.argsort(dist)[:n]
        return [self.data[i] for i in dist]
    