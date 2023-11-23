# Wow! This is sparta!
import faiss
from faiss import write_index, read_index
from sentence_transformers import SentenceTransformer
import math
import pandas as pd
import requests
from urllib.parse import urlencode
import os
import numpy as np
from catboost import CatBoostRanker, Pool
from functools import partial
import json
import pyarrow.parquet as pq
import time

start_time = time.time()

seed = 42
np.random.seed(seed)

candidates = pd.read_parquet(
    'videos.parquet',
    engine='fastparquet',
    columns=['video_id', 'video_title']
)

candidates = candidates.sample(n=7_000_000, replace=False, random_state=seed)

corpus = candidates['video_title'].apply(lambda x: x.lower()).values
video_ids = candidates['video_id'].values
del candidates

st_model = SentenceTransformer(
    'cointegrated/rubert-tiny2',
    device='cuda'
)

# когда прогоните один раз у вас на диске уже будет сохранен faiss индекс
# можно поставить значение True, чтобы сэкономить время на формирование индекса
use_formed_index = False

d = 312
if not use_formed_index:
    cpu_index = faiss.IndexFlatL2(d)
    cpu_index.is_trained, cpu_index.ntotal

# если уже есть файл ind2videoid для вашего faiss индекса - True
use_formed_id_mapping = False

if not use_formed_id_mapping:
    ind2videoid = {ind: video_id for ind, video_id in enumerate(video_ids)}
    with open('ind2videoid.json', 'w+') as f:
        json.dump(ind2videoid, f, indent=4)
else:
    with open('ind2videoid.json', 'r') as f:
        ind2videoid = json.load(f)

batch_size = 100000
num_batches = math.ceil(len(corpus) / batch_size)

if not use_formed_index:
    try:
        for i in range(num_batches):
            # формируем батч
            start, end = i * batch_size, (i + 1) * batch_size
            corpus_batch = corpus[start:end]

            # считаем вектора для всех предложений в батче
            embeddings = st_model.encode(
                corpus_batch,
                batch_size=1000,
                show_progress_bar=True
            )

            # добавляем новые батч векторов в индекс и сохраняем его
            cpu_index.add(embeddings)
            write_index(cpu_index, 'candidates.index')

            print(
                f'batch: {i + 1} / {num_batches}, vectors: {cpu_index.ntotal}')

            # чистим ОЗУ
            del embeddings
    except KeyboardInterrupt:
        print('Остановлено пользователем')
        try:
            del embeddings
        except:
            pass



if __name__ == '__main__':
    ...
