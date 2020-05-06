from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
import jieba
#特征抽取对于列表里的数据进行处理
def countervec():
    #文本特征抽取
    # 实例化CountVectorizer
    vertor = CountVectorizer()
    # 调用fit_transform输入并转换数据
    # res = vertor.fit_transform(["life is short is bag,i like python", "life is too short,i don't like c++"])
    res = vertor.fit_transform(["人生苦短,我喜欢python", "人生漫长,我不喜欢c++"])
    # 打印结果
    print(vertor.get_feature_names())
    #sparse矩阵转换为array
    print(res.toarray())
    print(type(res))
def hanzivec():
    """结合jieba分词对中文文本进行特征值化"""
    datas=["木兰之枻沙棠舟，玉箫金管坐两头。","美酒尊中置千斛，载妓随波任去留。","仙人有待乘黄鹤，海客无心随白鸥。",
        "屈平词赋悬日月，楚王台榭空山丘。","兴酣落笔摇五岳，诗成笑傲凌沧洲。","功名富贵若长在，汉水亦应西北流。"]
    ds=[]
    for data in datas:
        #使用jieba分词之后返回的是一个生成器，要先转换成列表再转换成字符串
        # print(" ".join(list(jieba.cut(data))))
       ds.append(" ".join(list(jieba.cut(data))))

    print(ds)

    cv=CountVectorizer()
    fd=cv.fit_transform(ds)
    print(fd.toarray())
    print(cv.get_feature_names())
def dictvec():
    """字典数据特征抽取，类别型，one-hot编码"""
    #实例化DictVectorizer
    city=[{"city":"北京","temperature":30},
        {"city":"上海","temperature":60},
        {"city":"深圳","temperature":20}
          ]
    #默认转化为Sparse矩阵，Sparse=False转化为数组
    di=DictVectorizer(sparse=False)
    #调用fit_transform
    data=di.fit_transform(city)
    #打印特征名
    print(di.get_feature_names())
    print(di.inverse_transform(data))
    print(data)
    return None
def run():
    # countervec()
    hanzivec()
    # dictvec()

if __name__=="__main__":
    run()