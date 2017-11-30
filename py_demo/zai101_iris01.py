#coding=utf-8
'''
Created on 2016.12.25
TopQuant-极宽量化系统·培训课件-配套教学python程序
@ www.TopQuant.vip      www.ziwang.com

'''

import pandas as pd

#-----------------------

#1 
fss='dat/iris.csv'
df=pd.read_csv(fss,index_col=False)
print('\n#1 df')
print(df.tail())
print(df.describe())

#2
d10=df['xname'].value_counts()
print('\n#2 xname')
print(d10)

#-----------------------
print('\nok!')
