from sklearn.cluster import SpectralClustering
import numpy as np
import math

X = np.array([[185.4, 72.6],
              [155.0, 54.4],
              [170.2, 99.9],
              [172.2, 97.3],
              [157.5, 59.0],
              [190.5, 81.6],
              [188.0, 77.1],
              [167.6, 97.3],
              [172.7, 93.3],
              [154.9, 59.0]])

w, h = 10, 10;

# construct similarity matrix, similarity of two smaple=100-Euclidean Distance
Matrix = [[100 - math.hypot(X[x][0] - X[y][0], X[x][1] - X[y][1]) for x in range(w)] for y in range(h)]

sc = SpectralClustering(3, affinity='precomputed', n_init=10)
sc.fit(Matrix)

print('spectral clustering')
print(sc.labels_)
