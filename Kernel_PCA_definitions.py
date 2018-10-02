__author__ = "rohil"


from sklearn.datasets.samples_generator import make_swiss_roll
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
import itertools
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA



def swiss_roll_dataset(number_of_samples=1000,plot=True):
    X, color = make_swiss_roll(n_samples=number_of_samples, random_state=123)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.rainbow)
    plt.title('Swiss Roll in 3D')
    if plot:
        plt.show()
    plt.clf()
    return X, color

def classic_pca(dataset,color,n_components=2, plot=True):
    scikit_pca = PCA(n_components=n_components)
    X_spca = scikit_pca.fit_transform(dataset)


    if n_components == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_spca[:, 0], X_spca[:, 1], c=color, cmap=plt.cm.rainbow)
        plt.title('First 2 principal components after Linear PCA')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
    if n_components == 1:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_spca, np.zeros((X_spca.shape[0], 1)), c=color.reshape((dataset.shape[0],1)), cmap=plt.cm.rainbow)
        plt.title('First principal component after Linear PCA')
        plt.xlabel('PC1')

    if plot:
        plt.show()
    plt.clf()

    return X_spca
def kernel_pca(X,  n_components=2,type_of_kernel= None,gamma=0.1,poly_constant=1,sigmoid_constant=1,sigmoid_alpha=0.1):
    number_of_features = X.shape[1]
    if type_of_kernel == 'rbf':
        # Calculating the squared Euclidean distances for every pair of points
        # in the MxN dimensional dataset.
        sq_dists = pdist(X, 'sqeuclidean')

        # Converting the pairwise distances into a symmetric MxM matrix.
        mat_sq_dists = squareform(sq_dists)
        # Computing the MxM kernel matrix.
        K = exp(-gamma * mat_sq_dists)

    if type_of_kernel == 'poly':
        K=[]
        for combinations in itertools.combinations(X, 2):
            k= (combinations[0].T.dot(combinations[1])+poly_constant)**number_of_features

            K.append(k)
        K=np.array(K)

        K = squareform(K)

    if type_of_kernel == 'sigmoid':
        K=[]
        for combinations in itertools.combinations(X, 2):
            k= np.tanh((sigmoid_alpha*(combinations[0].dot(combinations[1].T))+sigmoid_constant))
            K.append(k)
        K=np.array(K)
        K=squareform(K)
    # Centering the symmetric NxN kernel matrix.

    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenvalues in descending order with corresponding
    # eigenvectors from the symmetric matrix.
    eigvals, eigvecs = eigh(K)

    # Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
    X_pc = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))

    return X_pc


def plot_graph(data,plot_for_rbf=False,plot_for_sigmoid=False,plot_for_poly=False,plot=True):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=color, cmap=plt.cm.rainbow)
    if plot_for_rbf:
        plt.title('First 2 principal components after RBF Kernel PCA')
        plt.text(-0.14, 0.14, 'gamma = 0.1', fontsize=12)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
    if plot_for_sigmoid:
        plt.title('First 2 principal components after Sigmoid Kernel PCA')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
    if plot_for_poly:
        plt.title('First 2 principal components after Sigmoid Kernel PCA')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
    if plot:
        plt.show()



