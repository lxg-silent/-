import pandas as pd
df1=pd.DataFrame({
'a':[0,0,0,0,1,1,1,1],
'b':[0,0,1,1,0,0,1,1] ,
'c':[1,2,3,4,5,6,7,8],
'd':['h','i','j','k','l','m','n','o']
})
print(df1)
data=df1.iloc(1)
print(data)

quit()
df2=pd.DataFrame({
'a':[0,0,1,1],
'b':[0,1,0,1] ,
'e':[4,5,6,7],
'f':['x','y','z','z']
})
df = df1.merge(df2,on=['a','b'])
