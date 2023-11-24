import os
from typing import Callable
import pyarrow.parquet as pq
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from time import process_time
import pickle

logging.basicConfig(level=logging.INFO)


def _timeit(func: Callable):
    """Декоратор для замера времени выполнения функции"""

    def wrapper(*args, **kwargs):
        print(f'Выполняется функция: {func.__qualname__}...')
        start_time = process_time()
        result = func(*args, **kwargs)
        end_time = process_time()
        print(
            f"Время выполнения функции '{func.__qualname__}': {end_time - start_time} секунд"
        )
        return result

    return wrapper

class Engine:
    @_timeit
    def __init__(self, videos_file: str, features_file: str, model_path: str = 'vectorizer.pk'):
        self.vectorizer: TfidfVectorizer = TfidfVectorizer()
        self.videos = pd.read_parquet(videos_file, engine='fastparquet', columns=[
            'video_id', 'video_title', 'v_pub_datetime', 'channel_title'])
        # self.features = pd.read_parquet(features_file, engine='fastparquet', columns=[
        #                                 'video_id', 'v_likes', 'v_dislikes', 'v_category', 'v_pub_datetime'])
        self.vectors = None
        self.model_path = model_path
        self.fitted = False
        print(40)
        self.videos['video_title'] = self.videos['video_title'].apply(
            lambda x: x.lower())
        self.videos['channel_title'] = self.videos['channel_title'].apply(
            lambda x: x.lower())
        print(45)
        if os.path.exists(model_path):
            with open(model_path, 'rb') as fin:
                self.vectorizer: TfidfVectorizer = pickle.load(fin)
                self.vectors = self.vectorizer.transform(
                    self.videos[['video_title', 'channel_title']])
                self.fitted = True
        print(52)

    @_timeit
    def search_by_vector(self, query):
        try:
            query_vector = self.vectorizer.transform([query])
            data_chunk = self.videos.copy(True)

            similarities = cosine_similarity(
                query_vector, self.vectors).flatten()

            data_chunk['relevance'] = similarities

            relevant_chunk = data_chunk[data_chunk['relevance'] > 0.6]
            return relevant_chunk

        except Exception as e:
            logging.error(f"Ошибка при первичном поиске данных: {e}")
            return pd.DataFrame()

    @_timeit
    def fit(self):
        self.vectorizer = self.vectorizer.fit(
            self.videos[['video_title', 'channel_title']])
        self.vectors = self.vectorizer.transform(self.videos[['video_title', 'channel_title']])
        self.fitted = True
        self.save(self.model_path)

    def save(self, path):
        if self.fitted:
            with open(path, 'wb') as file:
                pickle.dump(self.vectorizer, file)
                self.fitted = True

    @_timeit
    def search(self, query: str):
        query = query.lower()
        if not self.fitted:
            self.fit()
        relevant_videos = self.search_by_vector(
            query).sort_values('relevance', ascending=False)

        return relevant_videos


if __name__ == "__main__":
    try:
        path = os.path.join(os.getcwd(), 'videos.parquet')
        app = Engine(videos_file=path, features_file='')
        working = True
        while working:
            query = input("Введите запрос: ")
            if (query in ('exit', 'выход')):
                working = False
                continue
            res = app.search(query)
            print('Результаты поиска: ')
            print(res.head(10))

    except Exception as e:
        logging.error(f"Ошибка при запуске приложения: {e}")


# 1)допилить на хэшмапы videos.parquet
# 2)реализовать релевантность с этими хэшами:
#   1.по названию
#   2.по дате и жанру
# 3)создать хэшмапу для features и по ней тоже релевантность
# 4)дать веса каждому элементу, влияющему на релевантность и вывести формулу
# 5)оформить вывод
