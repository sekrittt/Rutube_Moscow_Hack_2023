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

if not use_formed_index:
    del cpu_index

automarkup = pd.read_parquet(
    'automarkup.parquet',
    engine='fastparquet'
)

automarkup = automarkup[~automarkup['query'].isna()]
automarkup['query'] = automarkup['query'].apply(lambda x: x.lower())

n = 1000
top_n = automarkup['query'].value_counts()[:int(2*n)].index.to_list()
other = np.array(automarkup['query'].value_counts()[int(2*n):].index.to_list())
random_n = np.random.choice(other, size=n, replace=False).tolist()
queries = top_n + random_n
query2ind = {q: i for i, q in enumerate(queries)}

# когда прогоните один раз у вас на диске кандиды уже будут сохранены
# можете поставить значение True, чтобы сэкономить время
use_formed_candidates = False

qembeddings = st_model.encode(
    queries,
    batch_size=1000,
    show_progress_bar=True
)

search_cpu_index = read_index('candidates.index')

generated_cand_name = 'generated_candidates.parquet'
if not use_formed_candidates:
    topk = 300
    distance, faiss_ind = search_cpu_index.search(qembeddings, topk)

    generated_cand = {
        'query': [],
        'video_id': []
    }

    for i, q in enumerate(queries):
        vids = faiss_ind[i]
        generated_cand['video_id'] += [ind2videoid[v] for v in vids]
        generated_cand['query'] += [q] * len(vids)

    generated_cand = pd.DataFrame(generated_cand)

    generated_cand.to_parquet(
        generated_cand_name,
        engine='fastparquet'
    )
else:
    generated_cand = pd.read_parquet(
        generated_cand_name,
        engine='fastparquet'
    )

automarkup['target'] = [1] * automarkup.shape[0]
candidates_with_target = generated_cand.merge(
    automarkup[['query', 'video_id', 'target']],
    how='left',
    left_on=['query', 'video_id'],
    right_on=['query', 'video_id']
)
candidates_with_target['target'] = candidates_with_target['target'].fillna(0)

# очищаем ОЗУ
del generated_cand
del search_cpu_index
del automarkup
del st_model
del qembeddings

# здесь мы используем признаки актуальные на состояние 2-ого мая 2023 года
# вы можете использовать признаки наиболее актуальные для вашего кандидата

features_parquet = pq.ParquetFile('features.parquet')
features, filter_date = None, '2023-11-20'

for batch in features_parquet.iter_batches():
    tmp = batch.to_pandas()
    if features is None:
        features = tmp[tmp['report_date'] == filter_date]
    else:
        features = pd.concat([
            features,
            tmp[tmp['report_date'] == filter_date]
        ], axis=0)

# для baseline выбросим категориальные признаки и datetime признаки
# в своем решении вы сами можете решить использовать их или нет
features = features.drop(
    [
        'v_channel_reg_datetime',
        'v_channel_type',
        'v_category',
        'v_pub_datetime'
    ],
    axis=1
)

full_df = candidates_with_target.merge(
    features,
    how='inner',
    left_on='video_id',
    right_on='video_id'
)
del features
full_df = full_df.drop('report_date', axis=1)
full_df = full_df.drop_duplicates()

groups_to_drop = []
full_df['group_id'] = full_df.groupby(['query']).ngroup()
for group in full_df['group_id'].unique():
    part_df = full_df[full_df['group_id'] == group]
    target_sum = part_df['target'].values.sum()
    if target_sum <= 0:
        groups_to_drop += [group]
full_df = full_df[~full_df['group_id'].isin(groups_to_drop)]

groups = pd.Series(full_df['group_id'].unique())
permutation = groups.sample(frac=1, random_state=seed)
train_groups, val_groups, test_groups = np.split(
    permutation,
    [int(0.75 * len(permutation)), int(0.90 * len(permutation))]
)

train_df = full_df[full_df['group_id'].isin(train_groups)]
val_df = full_df[full_df['group_id'].isin(val_groups)]
test_df = full_df[full_df['group_id'].isin(test_groups)]

train_df = train_df.sort_values('group_id')
val_df = val_df.sort_values('group_id')
test_df = test_df.sort_values('group_id')

metainfo_columns = ['query', 'video_id', 'target', 'group_id']

X_train = train_df.drop(metainfo_columns, axis=1)
y_train, g_train = train_df['target'], train_df['group_id']

X_val = val_df.drop(metainfo_columns, axis=1)
y_val, g_val = val_df['target'], val_df['group_id']

X_test = test_df.drop(metainfo_columns, axis=1)
y_test, g_test = test_df['target'], test_df['group_id']

train = Pool(
    data=X_train.values,
    label=y_train.values,
    group_id=g_train.values,
    feature_names=X_train.columns.to_list()
)

val = Pool(
    data=X_val.values,
    label=y_val.values,
    group_id=g_val.values,
    feature_names=X_val.columns.to_list()
)

test = Pool(
    data=X_test.values,
    label=y_test.values,
    group_id=g_test.values,
    feature_names=X_test.columns.to_list()
)

task_type = 'GPU'
metric_period = 250

parameters = {
    'task_type': task_type,
    'verbose': False,
    'random_seed': seed,
    'loss_function': 'QueryRMSE',
    'learning_rate': 0.001,
    'l2_leaf_reg': 30,
    'iterations': 4000,
    'max_depth': 3,
}

model = CatBoostRanker(**parameters)
model = model.fit(
    train,
    eval_set=val,
    plot=True,
    use_best_model=True,
    metric_period=metric_period
)
model.save_model('ranker.ckpt')


def _metrics_at(at, model, pool, metric='NDCG'):
    metric = metric + f':top={at}'
    eval_metrics = model.eval_metrics(pool, metrics=[metric])
    best_metrics = {}
    for key in eval_metrics.keys():
        best_metrics[key] = eval_metrics[key][model.best_iteration_]
    return best_metrics


metrics_train_at = partial(
    _metrics_at,
    model=model,
    pool=train
)

metrics_val_at = partial(
    _metrics_at,
    model=model,
    pool=val
)

metrics_test_at = partial(
    _metrics_at,
    model=model,
    pool=test
)

print(metrics_train_at(1), metrics_train_at(5),
      '\n', metrics_val_at(1), metrics_val_at(5),
      '\n', metrics_test_at(1), metrics_test_at(5))

# очищаем ОЗУ
del model
del full_df
del train_df, val_df, test_df
del train, val, test
del X_train, y_train, g_train
del X_val, y_val, g_val
del X_test, y_test, g_test

end_time = time.time()
total_time = (end_time - start_time) / 60
print(f'Общее время работы baseline: {int(total_time)} минут')
