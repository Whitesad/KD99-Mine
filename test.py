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

if __name__=='__main__':
    csv_data=pd.read_csv('./data/KDD99_train_set_origin.csv')
    print(csv_data.shape)

    pca=PCA(n_components=1)
    X=np.array(csv_data.values)
    tag=X[:,41]
    X=np.delete(X,-1,axis=1)

    new_X=pca.fit_transform(X)

    # plt.scatter(new_X[:,0],r=tag)
    # plt.show()

    # x,y,z=[],[],[]
    # for i in range(len(new_X)):
    #     x.append(new_X[i][0])
    #     y.append(new_X[i][1])
    #     # z.append(new_X[i][3])
    # plt.scatter(x,y,c=tag)
    # plt.show()

    # data_X,data_Y,data_Z=[[]*30],[[]*30],[[]*30]
    # for i in range(len(X)):
    #     data_X[tag[i]].append(X[i][0])
    #     data_Y[tag[i]].append(X[i][1])
    #     data_Z[tag[i]].append(X[i][2])

    # fig = plt.figure()
    # ax=fig.add_subplot(projection='3d')
    # ax.scatter(new_X[:, 0], new_X[:, 1], new_X[:, 2],c=tag)
    # plt.show()
