from sklearn import  datasets

# print(datasets.__dict__)
# print(datasets.__dict__.keys())

for key in datasets.__dict__.keys():
     if key.startswith("fetch"):
        try:
            datasets.__dict__[key](data_home=r"D:\Learning_Materials\scikit_learn_data\fetch下载的数据")
        except ConnectionResetError as e:
          print("出错了")
