from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
li=datasets.load_iris()
# print("特征值")
# print(li.data)
# print("目标值")
# print(li.target)
# train_test_split(特征值，目标值，test_size(测试集的大小，为浮点数)))
x_train,x_test,y_train,y_test=train_test_split(li.data,li.target,test_size=0.25)
#返回值，训练集 x_train y_train  测试集 x_test y_test
#x_train  训练集的特征值
#y_train  训练集的目标值
#x_test   测试集的特征集
#y_test   测试值的目标值