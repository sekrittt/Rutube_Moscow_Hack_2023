import os
import pyarrow.parquet as pq
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import json
import string


class IndexedData:
    @staticmethod
    def clean_text(text: str):
        # Удаляем все символы, кроме букв и цифр
        return text.translate(str.maketrans('', '', string.punctuation))

    @staticmethod
    def create(*, path2save: str, path2data: str):
        path = os.path.join(os.getcwd(), path2data)
        videos = pd.read_parquet(path, engine='fastparquet', columns=[
                                'video_id', 'video_title', 'channel_title'])

        # Инициализируем словарь для хранения результата
        word_video_dict = defaultdict(list)

        # Используем tqdm для отслеживания прогресса
        for index, row in tqdm(videos.iterrows(), total=len(videos), desc="Processing videos"):
            video_id = row['video_id']
            # Извлекаем числовой идентификатор из строки 'video_...'
            video_id_numeric = int(video_id.split('_')[1])

            video_title = row['video_title']
            channel_title = row['channel_title']

            # Очищаем текст от лишних символов
            video_title_clean = IndexedData.clean_text(video_title)
            channel_title_clean = IndexedData.clean_text(channel_title)

            # Разбиваем очищенный заголовок видео на слова
            words_video = video_title_clean.split()

            # Разбиваем очищенный заголовок канала на слова
            words_channel = channel_title_clean.split()

            # Приводим слова к нижнему регистру и добавляем видео в список для каждого слова
            for word in [*words_video, *words_channel]:
                # for word in words:
                word_lower = word.lower()
                word_video_dict[word_lower].append(video_id_numeric)
            # print(word_video_dict)

        # Убираем дубликаты и преобразуем defaultdict в обычный dict
        result_dict = {word: list(set(video_ids))
                    for word, video_ids in word_video_dict.items()}

        # Сохраняем результат в JSON файл
        with open(path2save, 'w', encoding='utf-8') as json_file:
            json.dump(result_dict, json_file, ensure_ascii=False, indent=4)

        print(f"Результат сохранен в файл: {path2save}")

if __name__ == '__main__':
    IndexedData.create(path2data='videos.parquet', path2save='result.json')
