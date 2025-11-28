from typing import List

class TextPartition():
    def __init__(self, max_len: int=512, overlap_len: int=None):
        self.max_len = max_len
        self.overlap_len = max_len // 2 if overlap_len is None else overlap_len
        self.levels = ["\n\n", "\n", " ", ""]

    def _overlap_split(self, text: str, level: int, partition: List[str]):
        split = text.split(self.levels[level])
        i = 0
        part = []
        len_part = 0
        overlap = []
        len_overlap = 0
        while i < len(split):
            if len_overlap > 0:
                part.extend(overlap)
                len_part += len_overlap
            len_overlap = 0
            overlap = []
            while i < len(split) and len_part + len(split[i]) + len(self.levels[level]) <= self.max_len:
                part.append(split[i])
                len_part += len(split[i]) + len(self.levels[level])
                i += 1
            if len_part > 0:
                partition.append(self.levels[level].join(part))
            part = []
            len_part = 0
            if i < len(split):
                j = i - 1
                while j >= 0 and len_overlap + len(split[j]) + len(self.levels[level]) <= self.overlap_len:
                    overlap.insert(0, split[j])
                    len_overlap += len(split[j]) + len(self.levels[level])
                    j -= 1

    def _recursive_split(self, text: str, level: int, partition: List[str]):
        if level >= len(self.levels):
            return
        
        if level < 2:
            split = text.split(self.levels[level])

            for part in split:
                if len(part) <= self.max_len:
                    partition.append(part)
                else:
                    self._recursive_split(part, level + 1, partition)
        else:
            self._overlap_split(text, level, partition)

    def spilt(self, text: str):
        partition = []
        self._recursive_split(text, 0, partition)
        return partition