import pickle
from tqdm import tqdm
from datasets import Dataset
from src.utils import LSH, TextPartition, cosine_similarity

class EmbededDataset():
    def __init__(self, encoder, dataset: Dataset, partition: TextPartition):
        self.encoder = encoder
        self.dataset = dataset
        self.partition = partition
        self.data = []

        self._build()

    def _get_embeddings(self, query: str):
        split = self.partition.split(query)
        embeddings = self.encoder.encode(split)
        return embeddings

    def _build(self):
        for i, el in tqdm(enumerate(self.dataset), total=len(self.dataset)):
            query = "\n\n".join([el["name"], el["ingredients"], el["text"]])
            self.data.append(self._get_embeddings(query))

    def get_dist(self, query1, query2):
        emb1 = self._get_embeddings(query1)
        emb2 = self._get_embeddings(query2)
        dist = 0
        for e1 in emb1:
            for e2 in emb2:
                dist = max(dist, cosine_similarity(e1, e2))
        return dist

    def __getitem__(self, key):
        return self.data[key]
    
    def save(self, path: str="data/embeddings.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(path: str="data/embeddings.pkl"):
        with open(path, "rb") as f:
            embeddings = pickle.load(f)
        return embeddings

class DataBase():
    def __init__(self, embeded_dataset: EmbededDataset, L: int, k: int):
        self.embeded_dataset = embeded_dataset
        self.encoder = embeded_dataset.encoder
        self.dataset = embeded_dataset.dataset
        self.data = embeded_dataset.data
        self.lsh = LSH(L, k, embed_dim=self.encoder.get_sentence_embedding_dimension())
        self.partition = embeded_dataset.partition
        self.embedings_ids = {}
        self.text_ids = {}
        self.embedings_cnt = 0

        self._build()

    def _build(self):
        for i, embeddings in tqdm(enumerate(self.data), total=len(self.dataset)):
            for embedding in embeddings:
                self.embedings_ids[tuple(embedding)] = self.embedings_cnt
                self.text_ids[self.embedings_cnt] = i
                self.embedings_cnt += 1
                self.lsh.add(embedding)

    def find(self, query: str, k: int=1, get_dist: bool=False):
        split = self.partition.split(query)
        embeddings = self.encoder.encode(split)
        results = set()
        for embedding in embeddings:
            neighbours, dist = self.lsh.find(embedding, n=5, get_dist=True)
            for neighbour in neighbours:
                embedding_id = self.embedings_ids[tuple(neighbour)]
                text_id = self.text_ids[embedding_id]
                results.add(text_id)
        results = list(results)[:k]
        if get_dist:
            return [{**self.dataset[results[i]], 'dist': dist[i]} for i in range(len(results))]
        return [self.dataset[i] for i in results]
    
    def get_dist(self, query1, query2):
        return self.embeded_dataset.get_dist(query1, query2)