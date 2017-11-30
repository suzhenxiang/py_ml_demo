#coding=utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
import ztop_ai as zai
import numpy as np
from sklearn.linear_model import LinearRegression,LogisticRegression
import pickle #用来保存训练的模型库
#划分数据集，训练集与测试集的比例，默认比例是0.25:0.75
fss='dat/iris2.csv'
df=pd.read_csv(fss,index_col=False)
xlst,ysgn=['x1','x2','x3','x4'],'xid'
x,y= df[xlst],df[ysgn]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1,test_size=0.2)#test_size:代表测试集的占比

mx = LogisticRegression()
mx.fit(x_train,y_train)
dacc=mx.score(x_test, y_test)
#测试git
print('\n8# mx:mx_sum,kok:{0:.2f}%'.format(dacc))