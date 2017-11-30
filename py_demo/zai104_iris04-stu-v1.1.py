#coding=utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
import ztop_ai as zai
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle #用来保存训练的模型库
#划分数据集，训练集与测试集的比例，默认比例是0.25:0.75
fss='dat/iris2.csv'
df=pd.read_csv(fss,index_col=False)
xlst,ysgn=['x1','x2','x3','x4'],'xid'
x,y= df[xlst],df[ysgn]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1,test_size=0.2)#test_size:代表测试集的占比

#利用相应算法训练数据，得出训练模型
#线性回归
mx = LinearRegression();
mx.fit(x_train,y_train)

# ##保存训练的模型到制定的文件，以便进行下一次预测
# df9=x_test.copy()
# #利用模型预测，预测测试集
#
# y_pred = mx.predict(x_test.values)
# df9['y_predsr']=y_pred
# df9['y_test']=y_test
# df9['y_pred']=df9['y_predsr'].apply(np.round).astype(int)
#
# #计算算法的准确度
# dacc=zai.ai_acc_xed(df9,1,False)
print mx.predict(x_test)
dacc=mx.score(x_test, y_test)
print dacc
print('\n8# mx:mx_sum,kok:{0:.2f}%'.format(dacc))