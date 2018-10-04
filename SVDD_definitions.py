__author__ = "rohil"

import numpy as np
import itertools
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from cvxpy import *
import pandas as pd

def one_class_dataset_generator(n_samples= 100, n_features=2,number_of_outliers=20, random_state=222,plot=False):
    np.random.RandomState(seed=random_state)
    X = 0.3*np.random.randn(n_samples, n_features)
    X_outliers = np.random.uniform(low=-4, high=4, size=(number_of_outliers, n_features))
    X_dataset_array= np.concatenate([X,X_outliers],axis=0)

    if plot:
        plt.scatter(X_dataset_array[:,0],X_dataset_array[:,1])
        plt.show()
    X_dataset=pd.DataFrame(X_dataset_array)
    return X_dataset_array,X_dataset
def gram_matrix(one_class_dataset_array, product_array):
    for combination in itertools.combinations(one_class_dataset_array, 2):
        product = np.dot(combination[0], combination[1])
        product_array.append(product)
    product_array = np.array(product_array)
    gram_matrix = squareform(product_array)
    print(gram_matrix)
    return gram_matrix

def svdd(one_class_dataset_array,one_class_dataset,epochs=100,learning_rate=0.1,Constant_for_SVDD=0.1):
    product_array=[]
    gm = gram_matrix(one_class_dataset_array, product_array)
    product_array_having_self_multiplication=[]
    for idx,rows in one_class_dataset.iterrows():
        product= np.dot(rows,rows)
        print(product)
        product_array_having_self_multiplication.append(product)
    product_array_having_self_multiplication=np.array(product_array_having_self_multiplication)

    alpha= Variable(one_class_dataset.shape[0])

    constraint = [alpha[i] <= Constant_for_SVDD for i in range(one_class_dataset.shape[0])]
    constraint1= [alpha[i] >= 0 for i in range(one_class_dataset.shape[0])]
    constraint_f= constraint+constraint1
    objective_function= Maximize((alpha*product_array_having_self_multiplication)+np.sum(squareform(alpha.T*alpha*gm)))

    problem= Problem(objective_function,constraint_f)

    problem.solve()
    print("Problem Status: %s" % problem.status)
    print(alpha)



if __name__ == "__main__":
    X_array,X=one_class_dataset_generator(n_samples=50,number_of_outliers=0,plot=False)

    svdd(X_array,X)

