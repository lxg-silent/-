from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
import pandas as pd
pd.DatetimeIndex
import pandas as pd
#数据集来源：
# https://www.kaggle.com/c/facebook-v-predicting-check-ins/data
#分类问题
# 特征值:x,y坐标  定位准确性 时间(时间戳)
#目标值:入住位置的id
#时间戳进行(年，月，日，周，时、分、秒)，可以单独抽取某个字段作为新特征(比如年)
#标签过多，可以删除一些没代表性的标签，这里把少于指定签到人数的位置删除

def knncls():
    """
    K-近邻预测用户签到位置
    :return:
    """
    #读取数据
    # f=open(r"D:\Learning_Materials\sklearn_data\预测入住位置数据\train.csv","r")
    #
    # data=pd.read_csv(r"D:\Learning_Materials\sklearn_data\预测入住位置数据\train.csv",engine='python',iterator=True)
    # loop=True
    # chunksize=100000
    # chunks=[]
    # while loop:
    #     try:
    #         chunk=data.get_chunk(chunksize)
    #         chunks.append(chunk)
    #     except StopIteration:
    #         loop=False
    # data=pd.concat(chunks,ignore_index=True)

    data=pd.read_csv(r"D:\Learning_Materials\sklearn_data\预测入住位置数据\train.csv",engine='python',nrows=100000)
    # data=pd.read_csv(f,engine='python',iterator=True)

    #处理数据
    #1、缩小数据(可以不用缩小)，查询数据筛选
    data=data.query("x > 1.0 & x<1.25 & y>2.5 & y <2.75")
    #x > 1.0 & x<1.25 & y>2.5 & y <2.75表示筛选满足这个条件的数据
    #"&"表示并且
    #处理时间的数据,这里的时间数据为时间戳
    #unit="s"表示转化到秒，当然也可以转化到分等，更多查看源码
    time_value=pd.to_datetime(data["time"],unit="s")
    print(time_value)
    #把日期格式转化成字典格式(pd.DatetimeIndex)
    # _field_ops = ['year', 'month', 'day', 'hour', 'minute', 'second',
    #               'weekofyear', 'week', 'weekday', 'dayofweek',
    #               'dayofyear', 'quarter', 'days_in_month',
    #               'daysinmonth', 'microsecond',
    #               'nanosecond']
    time_value=pd.DatetimeIndex(time_value)
    #构造一些特征
    data.loc[:,"day"]=time_value.day
    # data["day"]=time_value.day
    # data.loc[data.index,"hour"]=time_value.hour
    data["hour"]=time_value.hour
    # data.loc[data.index,"weekday"]=time_value.weekday
    data["weekday"]=time_value.weekday
    #把时间戳特征(time)删除,axis=1删除列
    data.drop(["time"],axis=1)
    # print(data)
    #将签到位置少于n个用户的删除
    #pd.groupby分组,统计place_id(签到ID)的次数
    place_count=data.groupby("place_id").count()
    # print(place_count)
    #reset_index还原索引
    tf=place_count[place_count.row_id>=2].reset_index()
    print(tf)
    #isin判断某一个字段在某个数组里
    data =data[data["place_id"].isin(tf.place_id)]
    #取出数据当中的特征值和目标值，这里目标值为place_id
    y=data["place_id"]
    #取出目标值
    x=data.drop(["place_id","row_id"],axis=1)
    print(x)

    #删除目标值place_id,留下特征值,axis=1按列删除

    #进行数据的分割训练集合测试集
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

    #特征工程(标准化)
    std=StandardScaler()

    #对测试集和训练集的特征值进行标准化
    x_train =std.fit_transform(x_train)
    x_test=std.transform(x_test)

    #进行算法流程
    knn=KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train,y_train)

    #得出预测结果
    y_predict=knn.predict(x_test)

    print("预测的目标签到位置为:",y_predict)

    #得出准确率
    print(" 预测准确率：",knn.score(x_test,y_test))

if __name__=="__main__":
    knncls()
