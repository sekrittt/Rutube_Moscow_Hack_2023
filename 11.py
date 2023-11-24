import os
import pyarrow.parquet as pq
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from time import process_time

import pickle

logging.basicConfig(level=logging.INFO)


def _timeit(func):
    """Декоратор для замера времени выполнения функции"""

    def wrapper(*args, **kwargs):
        start_time = process_time()
        result = func(*args, **kwargs)
        end_time = process_time()
        print(
            f"Время выполнения функции '{func.__name__}': {end_time - start_time} секунд"
        )
        return result

    return wrapper

class Engine:

    def __init__(self, path: str):
        self.path = path
        self.relevant_videos = pd.DataFrame()
        self.vectorizer = TfidfVectorizer()
        self.fitted = False
        if os.path.exists('vectorizer.pk'):
            with open('vectorizer.pk', 'rb') as fin:
                self.vectorizer = pickle.load(fin)
                self.fitted = True
        self.data = pd.read_parquet(path, engine='fastparquet', columns=['video_id', 'video_title'])
        self.title_vectors = None

    @_timeit
    def primary_search(self, data_chunk, query):
        try:
            query_vector = self.vectorizer.transform([query])

            similarities = cosine_similarity(
                query_vector, self.title_vectors).flatten()

            data_chunk['relevance'] = similarities

            relevant_chunk = data_chunk[data_chunk['relevance'] > 0.6]
            return relevant_chunk[['video_id', 'video_title', 'relevance']]

        except Exception as e:
            logging.error(f"Ошибка при первичном поиске данных: {e}")
            return pd.DataFrame()

    @_timeit
    def fit(self, data_chunk):
        titles = data_chunk['video_title'].apply(lambda x: x.lower())
        self.vectorizer = self.vectorizer.fit(titles)
        self.title_vectors = self.vectorizer.transform(titles)

    def save(self):
        if not self.fitted:
            with open('vectorizer.pk', 'wb') as fin:
                pickle.dump(self.vectorizer, fin)
                self.fitted = True

    @_timeit
    def search(self, query: str):
        query = query.lower()



if __name__ == "__main__":
    try:
        path = os.path.join(os.getcwd(), 'videos.parquet')
        query = input("Введите запрос: ")
        app = Engine(path=path)

    except Exception as e:
        logging.error(f"Ошибка при запуске приложения: {e}")


# 1)допилить на хэшмапы videos.parquet
# 2)реализовать релевантность с этими хэшами:
#   1.по названию
#   2.по дате и жанру
# 3)создать хэшмапу для features и по ней тоже релевантность
# 4)дать веса каждому элементу, влияющему на релевантность и вывести формулу
# 5)оформить вывод
