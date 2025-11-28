import torch
import torch.nn.functional as F

from typing import List

class LSH():
    def __init__(self, L: int, k: int, embed_dim: int):
        self.L = L
        self.k = k
        self.embed_dim = embed_dim
        self.hash_functions = torch.rand(size=(self.L, self.k, embed_dim))
        self.data = [{} for _ in range(self.L)]

    def add(self, obj: torch.tensor) -> None:
        for i, hash_function in enumerate(self.hash_functions):
            hash = [0 for _ in range(self.k)]
            for j, v in enumerate(hash_function):
                if F.cosine_similarity(v, obj, dim=0).item() >= 0:
                    hash[j] = 1
            hash = tuple(hash)
            self.data[i][hash] = self.data[i].get(hash, [])
            self.data[i][hash].append(obj)

    def find(self, obj: torch.tensor, n: int=1) -> List[torch.tensor]:
        neighbours = []
        for i, hash_function in enumerate(self.hash_functions):
            hash = [0 for _ in range(self.k)]
            for j, v in enumerate(hash_function):
                if F.cosine_similarity(v, obj, dim=0).item() >= 0:
                    hash[j] = 1
            hash = tuple(hash)
            neighbours.extend(self.data[i].get(hash, []))
        dist = [tuple([-F.cosine_similarity(neighbour, obj, dim=0).item(), i]) for i, neighbour in enumerate(neighbours)]
        dist = sorted(dist)[:n]
        return [neighbours[i] for _, i in dist]