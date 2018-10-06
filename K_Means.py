__author__ = "zarria"

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
from sklearn.datasets.samples_generator import make_swiss_roll
from mpl_toolkits.mplot3d import Axes3D

# Definition of data importation
def rbf_kernel(data1, data2, sigma):
    euc_square = np.linalg.norm(np.subtract(data1, data2))**2
    return np.exp(-euc_square/2*sigma**2)

def polynomial_kernel(data1, data2, constant, degree):
    return (np.dot(data1, data2)+constant)**degree

def sigmoid_kernel(data1, data2, sigmoid_alpha, sigmoid_constant):
    return np.tanh(sigmoid_alpha*np.dot(data1, data2)+sigmoid_constant)


# Definition Kernels terms
def second_term(data, classifications, kernel_type=None, sigma=0.1, pol_constant=15, pol_degree=2, sig_alpha=0.1, sig_constant=1):
    result = 0

    for item in classifications:
        if kernel_type == "sig":
            result += sigmoid_kernel(data, item, sig_alpha, sig_constant)
        if kernel_type == "poly":
            result += polynomial_kernel(data, item, pol_constant, pol_degree)
        if kernel_type == "rbf":
            result += rbf_kernel(data, item, sigma)

    return result / len(classifications)

def third_term(centroids, kernel_type=None, sigma=0.1, pol_constant=15, pol_degree=2, sig_alpha=0.1, sig_constant=1):
    result = 0
    for i in range(len(centroids)):
        for j in range(len(centroids)):
            if kernel_type == "sig":
                result += sigmoid_kernel(centroids[i], centroids[j], sig_alpha, sig_constant)
            if kernel_type == "poly":
                result += polynomial_kernel(centroids[i], centroids[j], pol_constant, pol_degree)
            if kernel_type == "rbf":
                result += rbf_kernel(centroids[i], centroids[j], sigma)

    return result / len(centroids)


# K means class
class K_Means:

    def __init__(self, k=6, tol=0.001, max_iter=300, kernel_type=None, sigma=0.1, pol_constant=15, pol_degree=2, sig_alpha=0.1, sig_constant=1):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.pol_constant = pol_constant
        self.pol_degree = pol_degree
        self.sig_alpha = sig_alpha
        self.sig_constant = sig_constant

    def fit(self, data):

        self.centroids = {}

        # Centroids initialization
        element = random.sample(range(0, data.shape[0]), self.k)
        j = 0
        for i in element:
            self.centroids[j] = data[i]
            j += 1

        for i in range(self.max_iter):

            # Classification
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []
                self.classifications[i].append(self.centroids[i])

            dist3 = third_term(self.centroids, self.kernel_type)
            # Distance calculation using the 2 Norm distance
            for featureset in data:
                if self.kernel_type:

                    if self.kernel_type == "sig":
                        dist1 = sigmoid_kernel(
                            featureset, featureset, self.sig_alpha, self.sig_constant)
                    if self.kernel_type == "poly":
                        dist1 = polynomial_kernel(
                            featureset, featureset, self.pol_constant, self.pol_degree)
                    if self.kernel_type == "rbf":
                        dist1 = rbf_kernel(featureset, featureset, self.sigma)
                    dist2 = [second_term(featureset, self.classifications[i], self.kernel_type, self.sigma,
                                         self.pol_constant, self.pol_degree, self.sig_alpha, self.sig_constant) for i in range(self.k)]
                    distances = np.add(dist1, np.add(dist2, dist3))
                else:
                    distances = [np.linalg.norm(
                        featureset-self.centroids[centroid], ord=2) for centroid in self.centroids]
                classification = distances.tolist().index(min(distances))
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
                print(original_centroid, current_centroid)

                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    optimized = False

            if optimized:
                break
