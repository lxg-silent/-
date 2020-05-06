from sklearn.decomposition import PCA
import requests


def pca():
    """主成分分析进行特征降维"""
    #保留百分之90的特征
    pca=PCA(n_components=0.9)
    data=pca.fit_transform([[2,8,4,5],[6,3,0,8],[5,4,9,1]])
    print(data)
if __name__=="__main__":
    pca()