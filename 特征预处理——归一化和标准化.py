from sklearn.preprocessing import  MinMaxScaler,StandardScaler,Imputer
import numpy as np
def mm():
    """归一化处理（对二维数组）
    """
    #实例化MinMaxScaler
    data=np.arange(10,30).reshape(4,5)
    #feature_range指定值的范围，默认为0-1
    mm=MinMaxScaler(feature_range=[2,3])
    dm=mm.fit_transform(data)
    print(dm)
def ss():
    """
    标准化缩放
    :return:
    """
    data = np.arange(10, 30).reshape(4, 5)
    sd=StandardScaler()
    sf=sd.fit_transform(data)
    print(sf)

if __name__=="__main__":
    # mm()
    ss()