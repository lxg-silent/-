from  sklearn.datasets import  fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
#subset="all"下载所有内容
def nb():
    """朴素贝叶斯案例
    """
    news = fetch_20newsgroups(r"D:\Learning_Materials\sklearn_data\fetch下载的数据", subset="all")

    #进行数据分割，分割出训练集和测试集
    x_train,x_test,y_train,y_test=train_test_split(news.data,news.target,test_size=0.25)
    #
    tf=TfidfVectorizer()
    #以训练集当中的词的列表进行每篇文章重要性统计
    x_train=tf.fit_transform(x_train)
    print(tf.get_feature_names())
    #对测试集的词的重要性进行统计
    x_test=tf.transform(x_test)
    print(y_train)
    #进行朴素贝叶斯算法的预测
    mlt=MultinomialNB(alpha=1.0)
    mlt.fit(x_train,y_train)
    #预测结果
    y_predict=mlt.predict(x_test)
    print("预测的文章类别为:""",y_predict)
    #得出准确率
    print("准确率为:",mlt.score(x_test,y_test))
    print("每个类别的精确率和召回率",classification_report(y_test,y_predict,target_names=news.target_names))
if __name__=="__main__":
    nb()