from src import *

def nextStamp(tm, next_val=DELTA_MINUTES):
    fulldate = datetime.datetime(100, 1, 1, tm.hour, tm.minute, tm.second)
    fulldate = fulldate + datetime.timedelta(minutes=next_val)
    return fulldate.time()


def optimal_kmeans_dist(x, min_members=3, start=1, epochs=5):
  opt_model = (None, None, np.inf)
  for n_clusters in range(start, len(x.T)):
    stop = True
    for _ in range(epochs):
      labels, centroids, inertia = kmeans_min_members(data=x.T, k=n_clusters, min_members=min_members)
      if opt_model[2] > inertia:
        opt_model = (labels, centroids, inertia)
        stop = False
    if stop:
      break
  return opt_model


def kmeans_min_members(data, k, min_members, max_iterations=100):
    # Initialize centroids randomly
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for i in range(max_iterations):
        # Assign points to closest centroid
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = distances.argmin(axis=0)

        # Check if any cluster has less than min_members members
        for j in range(k):
            if (labels == j).sum() < min_members:
                # Merge with closest cluster
                distances[j, :] = np.inf
                closest_cluster = distances.argmin(axis=0)
                labels[labels == j] = closest_cluster[labels == j]

        # Compute inertia
        inertia = 0
        for j in range(k):
            cluster_data = data[labels == j]
            if len(cluster_data) > 0:
                cluster_centroid = cluster_data.mean(axis=0)
                centroids[j, :] = cluster_centroid
                inertia += ((cluster_data - cluster_centroid)**2).sum()

        # Check for convergence
        if i > 0 and np.all(labels == old_labels):
            break

        old_labels = labels.copy()

    return labels, centroids, inertia

