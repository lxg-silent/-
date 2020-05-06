import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def decision():
    """
    决策树对泰坦尼克号预测生死
    :return:
    """
    titan=pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")

    #处理数据，找出特征值和目标值
    x=titan[["pclass","age","sex"]]
    y=titan["survived"]
    #缺失值处理,inplace=True为替换
    x["age"].fillna(x["age"].mean(),inplace=True)
    #分割数据集到训练集测试集
    print(x)
    x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.25)

    #进行处理(特征工程) 特征-->> 类别-->> one_hot编码
    dict=DictVectorizer(sparse=False)
    #x_train.to_dict(orient='records')将每一行数据转化成字典
    x_train=dict.fit_transform(x_train.to_dict(orient='records'))
    #对测试集进行one_hot编码
    x_test=dict.transform(x_test.to_dict(orient='records'))
    #max_depth指定深度,min_samples_split参数表示减掉小于给定值的叶子，min_samples_leaf表示保留大于给定值的叶子
    # dec=DecisionTreeClassifier(max_depth=8,min_samples_leaf=1,min_samples_split=2)
    # dec.fit(x_train,y_train)
    # #预测准确率
    # print("预测的准确率:",dec.score(x_test,y_test))
    #
    # #导出决策树的结构
    # #tree.dot二进制文件
    # export_graphviz(dec,out_file="./tree.dot",feature_names=dict.feature_names_)
    # #自定义
    # export_graphviz(dec,out_file="./tree2.dot",feature_names=['年龄', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', '女', '男'])
    #随机森林的超参数调优
    rf =RandomForestClassifier()
    #自定义构造随机森林的超参数
    params={"n_estimators":[120,200,300,500,800,1200],"max_depth":[5,8,15,20,30]}
    #网格搜索与交叉验证
    gc=GridSearchCV(rf,param_grid=params,cv=10)
    gc.fit(x_train,y_train)
    print("准确率:",gc.score(x_test,y_test))
    print("查看选择的参数模型",gc.best_params_)
if __name__=="__main__":
    decision()