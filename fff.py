import os
import pyarrow.parquet as pq
parquet_file = pq.ParquetFile(os.path.join(
    os.getcwd(), 'features.parquet'))
    # os.getcwd(), 'videos.parquet'))

for i in parquet_file.iter_batches(batch_size=10):
    # print("RecordBatch")
    print(i)
    break
