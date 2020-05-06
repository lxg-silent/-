#https://archive.ics.uci.edu/ml/machine-learning-databases/
#数据来源:https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data
import  pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
def logistic():
    """
    逻辑回归做二分类进行癌症预测(根据细胞的属性特征)
    :return:
    """
    #构造列标签名字
    colunm=["Sample code number","Clump Thickness","Uniformity of Cell Size",
            "Uniformity of Cell Shape","Marginal Adhesion"
            ,"Single Epithelial Cell Size","Bare Nucle","Bland Chromatin",
            "Norrmal Nucleoli","Mitoses","Class"]
    #读取数据
    data=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",names=colunm,sep=",")
    print(data.head(10))
    print(data.columns)
    #缺失值进行处理,把问号替换成np.nan
    data=data.replace(to_replace="?",value=np.nan)
    #把缺失值删除,默认为行
    data=data.dropna()
    #进行数据的分割,按列，从0开始切片包左不包右，不要第一个和第是一个
    x_train,x_test,y_train,y_test=train_test_split(data[colunm[1:10]],data[colunm[10]],test_size=0.25)
    #进行标准化处理
    std=StandardScaler()
    x_train=std.fit_transform(x_train)
    x_test=std.transform(x_test)
    #逻辑回归预测
    lg=LogisticRegression(C=1.0)
    lg.fit(x_train,y_train)
    y_predict=lg.predict(x_test)
    print(lg.coef_)
    print("准确率",lg.score(x_test,y_test))
    #2——》良性，4-》恶性
    print("召回率",classification_report(y_test,y_predict,labels=[2,4]
          ,target_names=["良性","恶性"]))



if __name__=="__main__":
    logistic()