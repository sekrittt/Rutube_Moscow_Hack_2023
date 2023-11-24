import os
import pandas as pd
import pyarrow.parquet as pq
from pprint import pprint

import datetime

# parquet_file = pq.ParquetFile(os.path.join(
#     os.getcwd(), 'features.parquet'))
#     # os.getcwd(), 'videos.parquet'))

# for i in parquet_file.iter_batches(batch_size=1):
#     # print("RecordBatch")
#     pprint(i.to_pandas().to_dict())
#     break

d = pd.read_parquet(os.path.join(
    # os.getcwd(), 'features.parquet'), engine='fastparquet', columns=['video_id', 'v_likes', 'v_dislikes', 'v_category', 'v_pub_datetime'])
    os.getcwd(), 'videos.parquet'), engine='fastparquet')



# print(
#     datetime.datetime.utcnow().timestamp() - a['v_pub_datetime'][0].timestamp())

print(d.head(10)[['video_title', 'channel_title']])
