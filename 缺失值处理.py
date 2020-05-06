#pandas处理缺失值示例(缺失值应该为np.Nan,np.Nan的格式为float类型)
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
def SNan():
    """对于Serial对象
   丢弃带有NAN的所有项"""
    data = pd.Series([1, np.nan, 5, np.nan])
    d1=data.dropna()
    print(data)
    print("="*30)
    print(d1)

def DNan():
    """对于DataFrame对象
     丢弃带有NAN的行"""
    data = pd.DataFrame([[1, 5, 9, np.nan], [np.nan, 3, 7, np.nan], [6, np.nan, 2, np.nan]
                       , [np.nan, np.nan, np.nan, np.nan], [1, 2, 3, np.nan]])
    d1=data.dropna()
    #丢弃所有元素都是NAN的行
    d2=data.dropna(how="all")
    print(d2)
    #丢弃所有元素都是NAN的列
    d3 = data.dropna(how="all",axis=1)
    print(d3)
    #只保留至少有3个非NAN值的行
    d4=data.dropna(thresh=3)
    print(d4)
def FNan():
    data = pd.DataFrame([[1, 5, 9, np.nan], [np.nan, 3, 7, np.nan], [6, np.nan, 2, np.nan]
                            , [np.nan, np.nan, np.nan, np.nan], [1, 2, 3, np.nan]])
    print(data)
    #isnull	判断数据是否缺失
    # print(data.isnull)
    #notnull	isnull的否定式
    # print(data.notnull)
    #fillna( )以常数替换NAN值
    d1=data.fillna(0)
    # print(d1)
    #后向填充(取每列Nan的值作为填充)
    d2=data.fillna(method='ffill')
    print(d2)
    #后项填充且可以连续填充的最大数量为1
    d3=data.fillna(method='ffill', limit=1)
    print(d3)
def  QNan():
    """处理问号等字符串缺失值"""
    data = pd.DataFrame([[1, 5, 9, np.nan], [np.nan, "?", 7, np.nan], [6, np.nan, 2, np.nan]
                            , [np.nan, np.nan, np.nan, np.nan], [1, 2, 3, np.nan]])
    #先使用replace替换成np.nan，再进行fillna或dropna处理
    d1=data.replace("?",np.nan)
    print(d1)
def INan():
    """使用sklearn处理缺失值"""
    #missing_values可以为NaN或nan,strategy填补的策略这里为平均值,0为列，1为行
    im=Imputer(missing_values="NaN",strategy="mean",axis=0)
    data=im.fit_transform([[1,4],[5,np.nan],[10,19]])
    print(data)

if __name__=="__main__":
    # SNan()
    # DNan()
    # FNan()
    # QNan()
    INan()