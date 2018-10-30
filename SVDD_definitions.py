__author__ = "rohil"

import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
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

def gram_matrix(one_class_dataset_array, kernel_type=None, coeff_poly=1, degree_poly= 3, gamma_poly=None, gamma_rbf= None):
    if kernel_type== 'linear':
        gm= linear_kernel(one_class_dataset_array)
    if kernel_type== 'poly':
        gm= polynomial_kernel(one_class_dataset_array, coef0=coeff_poly, degree=degree_poly,gamma=gamma_poly)
    if kernel_type == 'rbf':
        gm = rbf_kernel(one_class_dataset_array, gamma=gamma_rbf)

    for i in range(gm.shape[0]):
        gm[i,i]= gm[i,i]+0.0001
    return gm

def svdd(one_class_dataset_array,one_class_dataset,Constant_for_SVDD=0.1, type_of_kernel=None,coeff_poly=1,
         degree_poly= 3, gamma_poly=None, gamma_rbf= None):
    product_array=[]
    gm = gram_matrix(one_class_dataset_array, kernel_type=type_of_kernel, coeff_poly=coeff_poly,degree_poly=degree_poly
                     ,gamma_poly=gamma_poly,gamma_rbf=gamma_rbf)


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

    on_circle = []
    for i in range(one_class_dataset_array.shape[0]):
        if 0.0001 < a[i] < Constant_for_SVDD - 0.01:
            on_circle.append(i)
    radius_list = list()
    third_term = np.matmul(alpha.value.T, np.matmul(gm, alpha.value))

    for k in on_circle:
        for i in range(one_class_dataset.shape[0]):
            temp_second_term = alpha.value[i] * gm[i, k]

        temp_radius = np.sqrt(gm[k, k] - 2 * temp_second_term + third_term)
        radius_list.append(temp_radius)
    radius_list.sort()

    print(radius_list)
    ax = plt.subplot(1, 1, 1)
    ax.scatter(one_class_dataset_array[:, 0], one_class_dataset_array[:, 1])
    if len(radius_list)==1:
        circle = plt.Circle(b, radius_list[0], color='r', fill=False)
        ax.add_artist(circle)
        ax.plot(b[0], b[1], "or")
        plt.show()
    if len(radius_list)==2:
        circle = plt.Circle(b, radius_list[0], color='r',fill=False)
        circle1= plt.Circle(b, radius_list[1], color='r',fill=False)
        ax.add_artist(circle)
        ax.add_artist(circle1)
        ax.plot(b[0], b[1], "or")
        plt.show()
    if len(radius_list)==3:
        circle = plt.Circle(b, radius_list[0], color='r',fill=False)
        circle1= plt.Circle(b, radius_list[1], color='r',fill=False)
        circle2= plt.Circle(b, radius_list[2], color='r', fill=False)
        ax.add_artist(circle)
        ax.add_artist(circle1)
        ax.add_artist(circle2)
        ax.plot(b[0], b[1], "or")
        plt.show()

    #Finding the Radius of Circle and plotting results

    # if len(on_circle)!=0:
    #     point_on_circle=one_class_dataset.loc[on_circle[0]]
    #     radius= distance.euclidean(point_on_circle,b)
    #     ax = plt.subplot(1, 1, 1)
    #     ax.scatter(one_class_dataset_array[:, 0], one_class_dataset_array[:, 1])
    #     circle=plt.Circle(b,radius,color='r',fill=False)
    #     ax.add_artist(circle)
    #     ax.plot(b[0], b[1], "or")
    #     plt.show()

    print('The number of support vectors are',len(on_circle))

if __name__ == "__main__":
    X_array,X=one_class_dataset_generator(n_samples=300,number_of_outliers=10,plot=True)


    svdd(X_array,X)

