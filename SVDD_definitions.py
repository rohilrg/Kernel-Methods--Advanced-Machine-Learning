__author__ = "rohil"

import numpy as np
import itertools
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from cvxpy import *
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy import stats

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
    # for combination in itertools.combinations(one_class_dataset_array, 2):
    #     product = np.dot(combination[0], combination[1])
    #     product_array.append(product)
    # product_array = np.array(product_array)
    # gram_matrix = squareform(product_array)

    # print(gram_matrix)
    gm= linear_kernel(one_class_dataset_array)
    for i in range(gm.shape[0]):
        gm[i,i]= gm[i,i]+0.0001
    return gm

def svdd(one_class_dataset_array,one_class_dataset,Constant_for_SVDD=0.1):
    product_array=[]
    gm = gram_matrix(one_class_dataset_array, product_array)

    product_array_having_self_multiplication=[]
    for idx,rows in one_class_dataset.iterrows():
        product= np.dot(rows,rows)

        product_array_having_self_multiplication.append(product)
    product_array_having_self_multiplication=np.array(product_array_having_self_multiplication)

    product_array_having_self_multiplication=product_array_having_self_multiplication.reshape(len(product_array_having_self_multiplication),1)

    alpha= Variable((one_class_dataset.shape[0],1))


    #print((alpha.T*product_array_having_self_multiplication).shape)

    #print(constraint2,'fuc')
    constraint2= [np.ones((1,one_class_dataset_array.shape[0]))*alpha==1]
    constraint = [alpha[i] <= Constant_for_SVDD for i in range(one_class_dataset.shape[0])]
    constraint1= [alpha[i] >= 0 for i in range(one_class_dataset.shape[0])]
    constraint_f= constraint1+constraint+constraint2
    product1_obj= alpha.T*np.diag(gm)

    #product2_obj= alpha.T*gm*alpha
    product2_obj= quad_form(alpha,Parameter(shape=gm.shape,value=gm, PSD=True))

    objective_function= Maximize(product1_obj-product2_obj)

    problem= Problem(objective_function,constraint_f)

    problem.solve()
    print("Problem Status: %s" % problem.status)
    a=alpha.value
    a= np.array(a).reshape((len(a),1))


    b=0
    for i in range(one_class_dataset_array.shape[0]):
        b+= a[i].T*one_class_dataset_array[i]



    ax = plt.subplot(1,1,1)
    ax.scatter(one_class_dataset_array[:, 0], one_class_dataset_array[:, 1])
    ax.plot(b[0],b[1],"or")
    plt.show()
    inside=[]
    on_circle=[]
    outside=[]
    for i in range(one_class_dataset_array.shape[0]):

        if a[i]<0.001:
            inside.append(a[i])
        if 0.001 <a[i] < 0.09:
            on_circle.append(a[i])
        if a[i] > 0.09:
            outside.append(a[i])

    print('The number of points inside:',len(inside),'on the circle is:',len(on_circle),'outside circle is:',len(outside))

if __name__ == "__main__":
    X_array,X=one_class_dataset_generator(n_samples=200,number_of_outliers=10,plot=True)


    svdd(X_array,X)

