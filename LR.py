__author__ = "zarria"

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_circles
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import collections
from sklearn.metrics.pairwise import rbf_kernel

x_train, y_train = make_circles(n_samples=400, factor=.3, noise=.05)


def sig(x):
    return 1/(1+np.exp(-x))


def logistic_loss(y, y_hat):
    return -np.mean(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))


def sci_class(x_train, y_train, gamma):

        K = rbf_kernel(x_train, gamma=gamma)
        m = len(y_train)

        model = LogisticRegression(solver='lbfgs')
        model = model.fit(K, y_train)
        predicts = model.predict(K)

        colors = ["g", "r"]

        for i in range(len(predicts)):
                plt.scatter(x_train[i, 0], x_train[i, 1],
                            c=colors[predicts[i]])

        plt.title("Scikit-learn classification")
        plt.show()


def rbf_k(x_train, y_train, gamma, lr, iteration):

        colors = ["g", "r"]
        K = []
        w = np.zeros((x_train.shape[0], 1))
        b = np.zeros((x_train.shape[0], 1))
        for i in range(x_train.shape[0]):
                somme = 0
                for j in range(x_train.shape[0]):
                        somme += np.exp(-gamma *
                                        np.linalg.norm(x_train[i]-x_train[j]))
                K.append(somme)

        K = np.asarray(K)

        for i in range(iteration):
                z = np.matmul(K, w) + b
                a = sig(z)
                loss = logistic_loss(y_train, a)
                dz = a - y_train
                dw = 1/len(y_train) * np.matmul(K.T, dz)
                db = np.sum(dz)
                w = w - lr*dw
                b = b - lr*db

        predicts = sig(z)[0]
        for i in range(len(predicts)):
                plt.scatter(x_train[i, 0], x_train[i, 1],
                            c=colors[int(predicts[i])])

        plt.title("Classification using RBF kernel")
        plt.show()


def pol_k(x_train, y_train, constant, degree, lr, iteration):
        colors = ["g", "r"]
        K = []
        w = np.zeros((x_train.shape[0], 1))
        b = np.zeros((x_train.shape[0], 1))

        for i in range(x_train.shape[0]):
                somme = 0
                for j in range(x_train.shape[0]):
                        somme += (np.dot(x_train[i],
                                         x_train[j])+constant)**degree
                K.append(somme)

        K = np.asarray(K)

        for i in range(iteration):
                z = np.matmul(K, w) + b
                a = sig(z)
                loss = logistic_loss(y_train, a)
                dz = a - y_train
                dw = 1/len(y_train) * np.matmul(K.T, dz)
                db = np.sum(dz)
                w = w - lr*dw
                b = b - lr*db

        predicts = sig(z)[0]
        for i in range(len(predicts)):
                plt.scatter(x_train[i, 0], x_train[i, 1],
                            c=colors[int(predicts[i])])

        plt.title("Classification using polynomial kernel")
        plt.show()


def pol_k(x_train, y_train, constant, degree, lr, iteration):
        colors = ["g", "r"]
        K = []
        w = np.zeros((x_train.shape[0], 1))
        b = np.zeros((x_train.shape[0], 1))

        for i in range(x_train.shape[0]):
                somme = 0
                for j in range(x_train.shape[0]):
                        somme += (np.dot(x_train[i],
                                         x_train[j])+constant)**degree
                K.append(somme)

        K = np.asarray(K)

        for i in range(iteration):
                z = np.matmul(K, w) + b
                a = sig(z)
                loss = logistic_loss(y_train, a)
                dz = a - y_train
                dw = 1/len(y_train) * np.matmul(K.T, dz)
                db = np.sum(dz)
                w = w - lr*dw
                b = b - lr*db

        predicts = sig(z)[0]
        for i in range(len(predicts)):
                plt.scatter(x_train[i, 0], x_train[i, 1],
                            c=colors[int(predicts[i])])

        plt.title("Classification using polynomial kernel")
        plt.show()


def sig_k(x_train, y_train, sig_constant, sig_alpha, lr, iteration):
        colors = ["g", "r"]
        K = []
        w = np.zeros((x_train.shape[0], 1))
        b = np.zeros((x_train.shape[0], 1))

        for i in range(x_train.shape[0]):
                somme = 0
                for j in range(x_train.shape[0]):
                        somme += np.tanh(sig_alpha *
                                         np.dot(x_train[i], x_train[j])+sig_constant)
                K.append(somme)

        K = np.asarray(K)

        for i in range(iteration):
                z = np.matmul(K, w) + b
                a = sig(z)
                loss = logistic_loss(y_train, a)
                dz = a - y_train
                dw = 1/len(y_train) * np.matmul(K.T, dz)
                db = np.sum(dz)
                w = w - lr*dw
                b = b - lr*db

        predicts = sig(z)[0]
        for i in range(len(predicts)):
                plt.scatter(x_train[i, 0], x_train[i, 1],
                            c=colors[int(predicts[i])])

        plt.title("Classification using sigmoid kernel")
        plt.show()


sci_class(x_train, y_train, 0.1)

rbf_k(x_train, y_train, 0.3, 0.1, 200)

pol_k(x_train, y_train, 10, 2, 0.1, 200)

sig_k(x_train, y_train, 5, 0.1, 0.1, 200)
