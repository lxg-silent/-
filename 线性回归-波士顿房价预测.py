from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def linear():
    """
    线性回归直接预测房子价格
    :return:
    """
    lb=load_boston()
    #分割数据集到训练集和测试集
    x_train,x_test,y_train,y_test=train_test_split(lb.data,lb.target,test_size=0.25)
    #进行标准化处理
    #必须对特征值和目标值进行标准化处理，实例化两个标准化API
    std_x=StandardScaler()
    x_train=std_x.fit_transform(x_train)
    x_test=std_x.transform(x_test)
    #目标值
    std_y=StandardScaler()
    #sklearn0.19版本传进的必须是二维数组reshape(-1,1)，由于样本数不知道，则直接填-1，目标值只有一个
    y_train=std_y.fit_transform(y_train.reshape(-1,1))
    y_test=std_y.transform(y_test.reshape(-1,1).reshape(-1,1))
    print(y_train)
    #estimator预测
    #正规方程求解方式预测结果
    lr=LinearRegression()
    lr.fit(x_train,y_train)
    print(lr.coef_)
    #预测测试集的房子价格
    y_predict=lr.predict(x_test)
    #转化回标准化之前的数据形式
    y_predict=std_y.inverse_transform(y_predict)
    print("正规方程预测每个房子的价格：",y_predict)
    print("正规方程的均方误差:",mean_squared_error(std_y.inverse_transform(y_test),y_predict))
    #梯度下降
    sgd=SGDRegressor()
    sgd.fit(x_train,y_train)
    print(sgd.coef_)
    #岭回归
    #alpha正则化力度
    rd=Ridge(alpha=1.0)
    rd.fit(x_train,y_train)
    print(rd.coef_)
if __name__=="__main__":
    linear()