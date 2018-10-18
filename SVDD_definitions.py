__author__ = "rohil"

import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.spatial import distance

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

    gm= linear_kernel(one_class_dataset_array)
    for i in range(gm.shape[0]):
        gm[i,i]= gm[i,i]+0.0001
    return gm

def svdd(one_class_dataset_array,one_class_dataset,Constant_for_SVDD=0.1):
    product_array=[]
    gm = gram_matrix(one_class_dataset_array, product_array)


    # Variable declaration
    alpha= Variable((one_class_dataset.shape[0],1))
    #Constraints
    constraint2 = [np.ones((1,one_class_dataset_array.shape[0]))*alpha==1]
    constraint = [alpha[i] <= Constant_for_SVDD for i in range(one_class_dataset.shape[0])]
    constraint1 = [alpha[i] >= 0 for i in range(one_class_dataset.shape[0])]
    constraint_f = constraint1+constraint+constraint2

    #Objective_Function
    product1_obj = alpha.T*np.diag(gm)
    product2_obj= quad_form(alpha,Parameter(shape=gm.shape,value=gm, PSD=True))

    objective_function= Maximize(product1_obj-product2_obj)

    problem= Problem(objective_function,constraint_f)

    problem.solve()
    print("Problem Status: %s" % problem.status)
    if problem.status =='optimal':
        a=alpha.value
        a= np.array(a).reshape((len(a),1))
    else:
        return
    # Finding the Center of the circle
    b=0
    for i in range(one_class_dataset_array.shape[0]):
        b+= a[i].T*one_class_dataset_array[i]

    #Finding the Radius of Circle and plotting results
    on_circle=[]
    for i in range(one_class_dataset_array.shape[0]):
        if 0.0001 <a[i] < Constant_for_SVDD-0.01:
            on_circle.append(i)
    if len(on_circle)!=0:
        point_on_circle=one_class_dataset.loc[on_circle[0]]
        radius= distance.euclidean(point_on_circle,b)
        ax = plt.subplot(1, 1, 1)
        ax.scatter(one_class_dataset_array[:, 0], one_class_dataset_array[:, 1])
        circle=plt.Circle(b,radius,color='r',fill=False)
        ax.add_artist(circle)
        ax.plot(b[0], b[1], "or")
        plt.show()

    print('The number of support vectors are',len(on_circle))

if __name__ == "__main__":
    X_array,X=one_class_dataset_generator(n_samples=300,number_of_outliers=10,plot=True)


    svdd(X_array,X)

