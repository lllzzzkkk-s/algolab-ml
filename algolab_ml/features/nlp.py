from __future__ import annotations
from typing import Iterable, List
from gensim.models import Word2Vec

def train_word2vec(sentences: Iterable[List[str]], vector_size: int=100, window: int=5, min_count: int=2, workers: int=4, sg: int=1):
    model = Word2Vec(sentences=list(sentences), vector_size=vector_size, window=window, min_count=min_count, workers=workers, sg=sg)
    return model
