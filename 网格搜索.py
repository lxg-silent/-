from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
def cv():
    """对k-近邻进行网格搜索
    找出最优k值"""
    #读取数据
    f=open(r"D:\Learning_Materials\sklearn_data\预测入住位置数据\train.csv","r")

    data=pd.read_csv(r"D:\Learning_Materials\sklearn_data\预测入住位置数据\train.csv",engine='python',iterator=True)
    loop=True
    chunksize=100000
    chunks=[]
    while loop:
        try:
            chunk=data.get_chunk(chunksize)
            chunks.append(chunk)
        except StopIteration:
            loop=False
    data=pd.concat(chunks,ignore_index=True)
    # 1、缩小数据(可以不用缩小)，查询数据筛选
    data = data.query("x > 1.0 & x<1.25 & y>2.5 & y <2.75")
    # x > 1.0 & x<1.25 & y>2.5 & y <2.75表示筛选满足这个条件的数据
    # "&"表示并且
    # 处理时间的数据,这里的时间数据为时间戳
    # unit="s"表示转化到秒，当然也可以转化到分等，更多查看源码
    time_value = pd.to_datetime(data["time"], unit="s")
    # 把日期格式转化成字典格式(pd.DatetimeIndex)
    # _field_ops = ['year', 'month', 'day', 'hour', 'minute', 'second',
    #               'weekofyear', 'week', 'weekday', 'dayofweek',
    #               'dayofyear', 'quarter', 'days_in_month',
    #               'daysinmonth', 'microsecond',
    #               'nanosecond']
    time_value = pd.DatetimeIndex(time_value)
    # 构造一些特征
    data.loc[:, "day"] = time_value.day
    # data["day"]=time_value.day
    # data.loc[data.index,"hour"]=time_value.hour
    data["hour"] = time_value.hour
    # data.loc[data.index,"weekday"]=time_value.weekday
    data["weekday"] = time_value.weekday
    # 把时间戳特征(time)删除,axis=1删除列
    data.drop(["time"], axis=1)
    # print(data)
    # 将签到位置少于n个用户的删除
    # pd.groupby分组,统计place_id(签到ID)的次数
    place_count = data.groupby("place_id").count()
    # print(place_count)
    # reset_index还原索引
    tf = place_count[place_count.row_id >= 2].reset_index()
    # isin判断某一个字段在某个数组里
    data = data[data["place_id"].isin(tf.place_id)]
    # 取出数据当中的特征值和目标值，这里目标值为place_id
    y = data["place_id"]
    # 取出目标值
    x = data.drop(["place_id", "row_id"], axis=1)

    # 删除目标值place_id,留下特征值,axis=1按列删除

    # 进行数据的分割训练集合测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    # 特征工程(标准化)
    std = StandardScaler()

    # 对测试集和训练集的特征值进行标准化
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)
    #这里不需要再指定k-值参数
    knn = KNeighborsClassifier()
    #构造一些参数的值进行搜索
    param={"n_neighbors":[3,5,10]}
    #进行网格搜索
    gc=GridSearchCV(knn,param_grid=param,cv=10)
    gc.fit(x_train,y_train)
    #预测准确率
    print("在测试集上准确率",gc.score(x_test,y_test))
    print("在交叉验证当中最好的结果",gc.best_score_)
    print("选择的最好模型是:",gc.best_estimator_)
    print("每个超参数每次交叉验证的结果:",gc.cv_results_)
if __name__=="__main__":
    cv()