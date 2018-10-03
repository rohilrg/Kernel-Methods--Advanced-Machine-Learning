import random
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import style


colors = 10*["g", "r", "c", "b", "k"]


X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])


def k_means(data,K, tolerance):
    
    iterations = 3000

    centroids = {}
    for i in range(K):
        centroids[i] = data[i]
    
    for i in range(iterations):
        classifications = {}
        for i in range(K):
            classifications[i] = []
        
        for item in data:
            distance = [np.linalg.norm(item-centroids[centroid]) for centroid in centroids]
            classification = distance.index(min(distance))
            classifications[classification].append(item)
        prev_centroid = dict(centroids)

        for classification in classifications:
            centroids[classification] = np.average(classifications[classification],axis=0)

        optimized = True

        for c in centroids:
            original_centroid = prev_centroid[c]
            current_centroid = centroids[c]
            if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > tolerance:
                print("optimize")
                optimized = False
        if optimized:
            break

    for centroid in centroids:
        plt.scatter(centroids[centroid][0],centroids[centroid][1],marker="o",color="k",s=150,linewidths=5)

    for classification in classifications:
        color = colors[classification]
        for item in data:
            plt.scatter(item[0],item[1],marker="x",color=color,s=150,linewidths=5)

    plt.show()

k_means(X,2,0.001)

