from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import os
import pyarrow.parquet as pq
import pandas as pd
from fuzzywuzzy import fuzz
import logging
from tqdm import tqdm  # Adding tqdm
import pickle
import traceback


class Engine:

    def __init__(self, path: str):
        self.path = path
        self.video_info_file = 'video_info.pkl'
        self.video_info = defaultdict(list)
        self.video_similarity = defaultdict(int)

    def load_data(self):
        logging.info("Функция load_data запущена")

        def process_batch(batch):
            df: pd.DataFrame = batch.to_pandas()
            # Разделение video_title на слова и создание отдельных ключей в словаре
            df['words'] = df['video_title'].apply(lambda x: x.split())
            df['vid'] = df['video_id'].apply(lambda x: int(x[6:]))
            df = df.explode('words')
            df.groupby('words')['vid'].apply(list)
            for word, vids in df.groupby('words')['vid']:
                self.video_info[word].extend(vids)

        logging.info("Загрузка данных из файла Parquet")
        parquet_file = pq.ParquetFile(self.path)
        columns = parquet_file.schema.names

        video_title_index = columns.index(
            'video_title') if 'video_title' in columns else None
        video_id_index = columns.index(
            'video_id') if 'video_id' in columns else None

        if video_title_index is not None and video_id_index is not None:
            with ThreadPoolExecutor() as executor:
                with tqdm(total=parquet_file.num_row_groups, desc='Loading Data') as pbar:
                    for batch in parquet_file.iter_batches(batch_size=20000):
                        executor.submit(process_batch, batch)
                        pbar.update(1)
            self.save_to_file()
        else:
            logging.error(
                "Ошибка: Невозможно найти 'video_title' и/или 'video_id' в столбцах файла Parquet.")

    def save_to_file(self):
        logging.info("Функция save_to_file запущена")
        with open(self.video_info_file, 'wb') as f:
            pickle.dump(self.video_info, f)

    def load_from_file(self):
        logging.info("Функция load_from_file запущена")
        if os.path.exists(self.video_info_file):
            with open(self.video_info_file, 'rb') as f:
                self.video_info = pickle.load(f)

    def search(self, query):
        logging.info("Функция search запущена")
        # Итерируемся по video_title и соответствующим video_id
        relevant_chunk = {'video_id': [], 'video_title': [], 'similarity': []}

        # Предварительно вычисляем пороговое значение
        threshold = 90

        for video_title, video_ids in tqdm(self.video_info.items(), desc='Processing Video Titles'):
            # Вычисляем сходство для каждого video_title
            similarity = fuzz.ratio(query, video_title)

            if similarity > threshold:
                relevant_chunk['video_id'].extend(video_ids)
                relevant_chunk['video_title'].append(video_title)
                relevant_chunk['similarity'].append(similarity)

        # Возвращаем результаты с релевантностью
        return relevant_chunk


    def start(self):
        logging.info("Функция start запущена")
        try:
            if not self.video_info:
                # Загружаем информацию о video_id и video_title, если еще не загружено
                self.load_from_file()

            if not self.video_info:
                # Если информации всё еще нет, загружаем из parquet-файла
                self.load_data()

            while True:
                query = input("Введите запрос (для выхода введите 'exit'): ")
                if query.lower() == 'exit':
                    break

                # Выполняем поиск напрямую по ключам в self.video_info
                relevant_chunk = self.search(query)

                if not all(len(value) == 0 for value in relevant_chunk.values()):
                    # Выводим результаты, только если есть релевантные результаты
                    print("Общие релевантные видео:")
                    for video_id, video_title, similarity in zip(
                            relevant_chunk['video_id'], relevant_chunk['video_title'], relevant_chunk['similarity']):
                        print(
                            f"{video_id}: {video_title} (Сходство: {similarity})")

        except Exception as e:
            logging.error(f"Ошибка при запуске движка: {e}")


if __name__ == "__main__":
    try:
        path = os.path.join(os.getcwd(), 'train_data', 'videos.parquet')
        app = Engine(path=path)
        app.start()
    except Exception as e:
        logging.error(f"Ошибка при запуске приложения: {e}")
