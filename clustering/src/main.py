import pandas as pd
import numpy as np
from cluster import KMeans

df = pd.read_csv('data/training.csv', index_col=['PID'])
data = np.array(df)
n_samples, _ = data.shape

kmeans = KMeans(display_log=True)
kmeans.fit(data)

out_df = pd.DataFrame(kmeans.labels_, columns=['category'])
out_df.index.name = 'id'
out_df.to_csv('data/answer.csv')