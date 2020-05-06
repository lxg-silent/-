from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

def tfidfvec():
    """tf-idf分析词的重要程度(文本分类)"""
    datas = ["木兰之枻沙棠舟，玉箫金管坐两头。", "美酒尊中置千斛，载妓随波任去留。", "仙人有待乘黄鹤，海客无心随白鸥。",
             "屈平词赋悬日月，楚王台榭空山丘。", "兴酣落笔摇五岳，诗成笑傲凌沧洲。", "功名富贵若长在，汉水亦应西北流。"]
    ds = []
    for data in datas:
        # 使用jieba分词之后返回的是一个生成器，要先转换成列表再转换成字符串
        # print(" ".join(list(jieba.cut(data))))
        ds.append(" ".join(list(jieba.cut(data))))
    # print(ds)
    tf=TfidfVectorizer()
    df=tf.fit_transform(ds)
    print(df.toarray())
    print(tf.get_feature_names())



if __name__=="__main__":
    tfidfvec()