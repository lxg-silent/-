"""products.csv ：商品信息（product_id，product_name，aisle_id，department_id）
  ordder_products_prior.csv:订单与商品信息（order_id，product_id，add_to_cart_order，reordered）
  orders.csv 用户的订单信息（order_id，user_id，eval_set，order_number，order_dow，order_hour_of_day，days_since_prior_order）
  aisles.csv:商品所属具体物品类别(aisle_id,aisle)"""
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
#读取四张表的数据
prior=pd.read_csv(r"E:\instacart-market-basket-analysis\order_products__prior.csv\order_products__prior.csv",nrows=50000)
# print(prior.columns)
# print(prior["product_id"])
# print(prior.dtypes)
# print("="*30)
products=pd.read_csv(r"E:\instacart-market-basket-analysis\products.csv\products.csv",nrows=50000)
# products["product_id"]=products["product_id"].astype('float64')
# print(products['product_id'])
# print(products.dtypes)
# print("="*30)
orders=pd.read_csv(r"E:\instacart-market-basket-analysis\orders.csv\orders.csv",nrows=50000)
# print(orders.dtypes)
# print("="*30)
aisles=pd.read_csv(r"E:\instacart-market-basket-analysis\aisles.csv\aisles.csv",nrows=50000)
# print(aisles.dtypes)
# print("="*30)
#需要根据相同的id键来和并表为一张表
#合并四张表到一张表(用户(样本)-物品类别(特征))
# _mg=pd.merge(prior,products,left_on="product_id",right_on="product_id")#on参数表示按键合并，需给出两张表的键
_mg=prior.merge(products,on=["product_id","product_id"])#on参数表示按键合并，需给出两张表的键
# print(_mg)
_mg=pd.merge(_mg,orders,on="order_id",how="inner")
mt=pd.merge(_mg,aisles,on="aisle_id",how="inner")
# print(mt)
#交叉表(特殊的分组表)
#建立行为用户id，列为物品id
#crosstab(行名，列名)
cross=pd.crosstab(mt["user_id"],mt["aisle"])
# print(cross)
#进行主成分分析
pca=PCA(n_components=0.9)
data=pca.fit_transform(cross)
print(data.shape)
#进行聚类
#假设用户一共分为四个类别
km=KMeans(n_clusters=4)
km.fit(data)
predict=km.predict(data)
print(predict)
#显示聚类结果
plt.figure(figsize=(10,10))
#建立四个颜色的列表
colored=["orange","green","blue","purple"]
colr=[colored[i] for i in predict]

plt.scatter(data[:,1],data[:,10],color=colr)
plt.xlabel("1")
plt.ylabel("10")
plt.show()
#评判聚类效果，轮廓系数
score=silhouette_score(data,predict)
print(score)
