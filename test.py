import os
import json
import pyarrow.parquet as pq
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import nltk

# Загрузка русских стоп-слов
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

# Путь к файлу Parquet
path = os.path.join(os.getcwd(), 'videos.parquet')

# Чтение данных из Parquet-файла
videos_df = pd.read_parquet(path, engine='fastparquet', columns=['video_id', 'video_title'])

# Создание словаря для хранения слов и видео
word_video_dict = defaultdict(list)

# Обработка каждого видео
for index, row in videos_df.iterrows():
    video_id = row['video_id']
    video_title = row['video_title']

    # Токенизация заголовка видео
    words = [word.lower() for word in word_tokenize(video_title, language='russian') if word.isalnum() and word.lower() not in stop_words]

    # Добавление слов в словарь
    for word in words:
        word_video_dict[word].append(video_id)

# Запись словаря в файл JSON
output_path = os.path.join(os.getcwd(), 'word_video_dict.json')
with open(output_path, 'w', encoding='utf-8') as json_file:
    json.dump(word_video_dict, json_file, ensure_ascii=False)

print(f"Словарь записан в файл: {output_path}")
