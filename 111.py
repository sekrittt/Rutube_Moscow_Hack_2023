from spellchecker import SpellChecker
import language_tool_python
import os
import pyarrow.parquet as pq
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from time import process_time


from jax import grad, jit, vmap

import pickle

logging.basicConfig(level=logging.INFO)


def vec(func):
    f = jit(func)

    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        return result

    return wrapper

def _timeit(func):
    """Декоратор для замера времени выполнения функции"""

    def wrapper(*args, **kwargs):
        global logging
        start_time = process_time()
        result = func(*args, **kwargs)
        end_time = process_time()
        if logging:
            print(
                f"Время выполнения функции '{func.__name__}': {end_time - start_time} секунд"
            )
        return result

    return wrapper


path = os.path.join(os.getcwd(), 'videos.parquet')

tool: language_tool_python.LanguageTool = language_tool_python.LanguageTool('ru-RU')
vectorizer: TfidfVectorizer = TfidfVectorizer()
fitted = False
if os.path.exists('vectorizer.pk'):
    with open('vectorizer.pk', 'rb') as fin:
        vectorizer = pickle.load(fin)
        fitted = True

def check_logic(text: str) -> str:
    global tool
    matches = tool.check(text)
    corrected_query = language_tool_python.utils.correct(
        text, matches) if matches else text
    return corrected_query

@vec
def primary_search(data_chunk: pd.DataFrame, query: str) -> pd.DataFrame:
    global vectorizer
    titles = data_chunk['video_title']
    if not fitted:
        vectorizer = vectorizer.fit(titles)
    query_vector = vectorizer.transform([query])
    title_vectors = vectorizer.transform(titles)

    similarities = cosine_similarity(
        query_vector, title_vectors).flatten()

    data_chunk['relevance'] = similarities

    relevant_chunk = data_chunk[data_chunk['relevance'] > 0.6]
    return relevant_chunk[['video_id', 'video_title', 'relevance']]


@vec
def search(q: str) -> pd.DataFrame:
    global path, check_logic
    # query: str = check_logic(q)
    query: str = q

    relevant_videos: pd.DataFrame = pd.DataFrame()
    parquet_file: pq.ParquetFile = pq.ParquetFile(path)
    for index, banch in enumerate(parquet_file.iter_batches(batch_size=500000, columns=['video_id', 'video_title', 'relevance'], use_pandas_metadata=True)):
        relevant_chunk = primary_search(banch.to_pandas(), query)
        relevant_videos = pd.concat(
            [relevant_videos, relevant_chunk])

    return relevant_videos.sort_values(
        'relevance', ascending=False)

def save():
    global fitted
    with open('vectorizer.pk', 'wb') as fin:
        pickle.dump(vectorizer, fin)
        fitted = True

if __name__ == "__main__":
    try:

        query = input("Введите запрос: ")
        res = search(query)
        save()

        print("Общие релевантные видео:")
        print(res)
    except Exception as e:
        logging.error(f"Ошибка при запуске приложения: {e}")


# 1)допилить на хэшмапы videos.parquet
# 2)реализовать релевантность с этими хэшами:
#   1.по названию
#   2.по дате и жанру
# 3)создать хэшмапу для features и по ней тоже релевантность
# 4)дать веса каждому элементу, влияющему на релевантность и вывести формулу
# 5)оформить вывод
