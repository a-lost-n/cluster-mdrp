from src import *

def nextStamp(tm):
    fulldate = datetime.datetime(100, 1, 1, tm.hour, tm.minute, tm.second)
    fulldate = fulldate + datetime.timedelta(minutes=DELTA_MINUTES)
    return fulldate.time()

def optimal_kmeans_dist(x, start=1, epochs=5):
  min_inertia = np.inf
  opt_model = None
  for n_clusters in range(start, len(x.T)):
    stop = True
    for epoch in range(epochs):
      km_cluster = KMeans(n_clusters=n_clusters, n_init='auto').fit(x.T)
      count = np.unique(km_cluster.labels_, return_counts = True)[1]
      if min_inertia > km_cluster.inertia_ and not (1 in count) and not (2 in count):
        min_inertia = km_cluster.inertia_
        opt_model = km_cluster
        stop = False
    if stop:
      break
  return opt_model

