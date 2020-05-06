from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
data=ss.fit_transform([[1,2,3],[4,5,6]])
print(data)
dat=ss.fit([[5,6,1],[7,8,9]])
print(dat)
#由于使用了fit里的计算的值，transform里的数据将按照fit的标准来计算

dat=dat.transform([[1,2,3],[4,5,6]])
print(dat)