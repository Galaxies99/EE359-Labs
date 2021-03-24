import pandas as pd
import numpy as np
from cluster import KMeans
import time

start = time.time()
df = pd.read_csv('data/data.csv', index_col=['PID'])
data = np.array(df)
n_samples, _ = data.shape

kmeans = KMeans(display_log=True, n_init=1)
kmeans.fit(data)

out_df = pd.DataFrame(kmeans.labels_, columns=['category'])
out_df.index.name = 'id'
out_df.to_csv('data/labels.csv')
print('Final time cost: ', time.time() - start)