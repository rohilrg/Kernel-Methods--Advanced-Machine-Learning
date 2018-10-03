import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
from sklearn.datasets.samples_generator import make_swiss_roll
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
import itertools
from scipy import exp


# Definition of data importation
def swiss_roll_dataset(number_of_samples=1000, plot=True):
    X, color = make_swiss_roll(n_samples=number_of_samples, random_state=123)
    if plot:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.rainbow)
        plt.title('Swiss Roll in 3D')
        plt.show()
        plt.clf()
    return X, color


def kernilize_data(data, type_of_kernel=None, gamma=0.1, poly_constant=1, sigmoid_constant=1, sigmoid_alpha=0.1):
    number_of_feature = data.shape[1]
    if type_of_kernel == 'rbf':
        sq_dists = pdist(data, "sqeuclidean")
        mat_sq_dists = squareform(sq_dists)
        K = exp(-gamma * mat_sq_dists)

    if type_of_kernel == 'poly':
        K = []
        for combinations in itertools.combinations(data, 2):
            k = (combinations[0].T.dot(combinations[1]) +
                 poly_constant)**number_of_feature
            K.append(k)
        K = np.array(K)
        K = squareform(K)

    if type_of_kernel == 'sigmoid':
        K = []
        for combinations in itertools.combinations(data, 2):
            k = np.tanh(
                (sigmoid_alpha*(combinations[0].dot(combinations[1].T))+sigmoid_constant))
            K.append(k)
        K = np.array(K)
        K = squareform(K)

    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    return K


class K_Means:

    def __init__(self, k=6, tol=0.001, max_iter=300, type_of_kernel=None, gamma=0.1, poly_constant=1, sigmoid_constant=1, sigmoid_alpha=0.1):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.type_of_kernel = type_of_kernel
        self.gamma = gamma
        self.poly_constant = poly_constant
        self.sigmoid_constant = sigmoid_constant
        self.sigmoid_alpha = sigmoid_alpha



    def fit(self, data):

        self.centroids = {}

        # Centroids initialization
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):

            # Classification
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            # Distance calculation using the 2 Norm distance
            for featureset in data:
                distances = [np.linalg.norm(
                    featureset-self.centroids[centroid], ord=2) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(
                    self.classifications[classification], axis=0)

            optimized = True

            # Centroids redifinition
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    #print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data-self.centroids[centroid])
                     for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


# Data Import
data, _ = swiss_roll_dataset(1000, False)

#data = kernilize_data(data,type_of_kernel='poly')
colors = 10*["g", "r", "c", "b", "k"]

clf = K_Means(k=4)
clf.fit(data)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
# Plot the data
for centroid in clf.centroids:
    #plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],                marker="o", color="k", s=150, linewidths=5)
    ax.scatter(clf.centroids[0], clf.centroids[1],
               clf.centroids[2], c="k", cmap=plt.cm.rainbow)
#print(clf.classifications)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:

        ax.scatter(featureset[0], featureset[1], featureset[2],
                   color=color)

plt.show()

data = kernilize_data(data,type_of_kernel='rbf')
clf = K_Means(k=4, type_of_kernel='rbf')
clf.fit(data)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
# Plot the data
for centroid in clf.centroids:
    #plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],                marker="o", color="k", s=150, linewidths=5)
    ax.scatter(clf.centroids[0], clf.centroids[1],
               clf.centroids[2], c="k", cmap=plt.cm.rainbow)
#print(clf.classifications)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:

        ax.scatter(featureset[0], featureset[1], featureset[2],
                   color=color)

plt.show()
