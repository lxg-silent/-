import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(10,10))#画板,定义一个10x10的画板
plt.scatter([60,72,80,83,90],[126,151.2,157.5,168,174.3])#画散点图
plt.show()
a=[[1,2,3,4],[5,6,7,8],[6,7,8,9]]
b=[2,3,4,5]
print(np.multiply(a,b))
print(np.dot(a,b))