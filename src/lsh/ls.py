import torch
import torch.nn.functional as F

from typing import List

class LS():
    def __init__(self):
        self.data = []

    def add(self, obj: torch.tensor) -> None:
        self.data.append(obj)

    def find(self, obj: torch.tensor, n: int=1) -> List[torch.tensor]:
        dist = [tuple([-F.cosine_similarity(neighbour, obj, dim=0).item(), i]) for i, neighbour in enumerate(self.data)]
        dist = sorted(dist)[:n]
        return [self.data[i] for _, i in dist]