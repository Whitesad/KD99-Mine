import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def test():
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(arr)
    arr=np.delete(arr, -1, axis=1)
    print(arr)
    input()

def PCA2D(X,tag):
    pca=PCA(n_components=2)
    new_X=pca.transform(X)
    x,y=[],[],[]
    for i in range(len(new_X)):
        x.append(new_X[i][0])
        y.append(new_X[i][1])
    plt.scatter(x,y,c=tag)
    plt.show()
def PCA3D(X,tag):
    pca=PCA(n_components=3)
    new_X=pca.transform(X)
    fig = plt.figure()
    ax=fig.add_subplot(projection='3d')
    ax.scatter(new_X[:, 0], new_X[:, 1], new_X[:, 2],c=tag)
    plt.show()
def preHandleData(X):
    X = np.delete(X, -1, axis=1)


if __name__=='__main__':
    csv_data=pd.read_csv('./data/KDD99_train_set_origin.csv')
    print(csv_data.shape)
    X=np.array(csv_data.values)
    tag=X[:,41]
    X=np.delete(X,-1,axis=1)

    PCA2D(X,tag)