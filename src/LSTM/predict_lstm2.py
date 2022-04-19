# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 09:51:04 2020

@author: 黄百万
"""
# 导入 keras 相关模块
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import csv
from matplotlib import pyplot as plt
import glob
import math
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split

'''用前22个人做训练，后8个人做预测'''
lstm_unit=128     #hidden layer units
time_step=10
#batch_size=32    #每一批次训练多少个样例
#input_size=1      #输入层维度
#output_size=1     #输出层维度
lr=0.0006         #学习率
data_x,data_y=[],[]   #训练集，x存储时间步长的数据，y存储标签
data_dim=1
num_classes = 72
one_hot=[]

def series_to_supervised(data):
    for i in range(len(data)-time_step-1):        
        x=data[i:i+time_step]
        x=np.array(x)
        x=x.reshape(time_step,1)#一维变二维
        y=data[i+time_step:i+time_step+1]
        data_x.append(x.tolist())#训练数据
        data_y.append(y)#标签


# dataset dir    
csvx_list=glob.glob(r"E:/datasets/viewpoint/Gaze_txt_files/Gaze_txt_files/*/179.*.txt")

for filename in csvx_list:
    x_i=[]
    y_i=[]
    frame=[]
    with open(filename,mode='r', encoding='utf-8') as f:# 打开txt文件，以‘utf-8'编码读
        lines = f.readlines()  # 以行的形式进行读取文件,line存储所有行
        for line in lines:
            a=line.split(",")
            b = float(a[6])  # 这是选取需要读取的位数
            b = int(12*b)#获取长边块
            d = float(a[7])
            d=int(6*d)#获取短边块
            tile=12*d+b
            x_i.append(tile) # 将其添加在列表之中
            
    m=[x_i[i] for i in range(0,len(x_i),5)]
    series_to_supervised(m)
    

# sample rate
for i in range(len(data_y)):
    one_hot_y = np.zeros(num_classes)
    a=data_y[i]
    one_hot_y[a[0]]= 1
    one_hot.append(one_hot_y.tolist())

data_y=one_hot
    

#train_x=data_x[:6269]
#test_x=data_x[6270:]
#train_y=data_y[:6269]
#test_y=data_y[6270:]

data_x=np.array(data_x)
data_y=np.array(data_y)

train_x, test_x, train_y, test_y = train_test_split(data_x,data_y, test_size=0.2, random_state=33)

model = Sequential()
model.add(LSTM(lstm_unit, return_sequences=True,
               input_shape=(time_step, data_dim)))    
model.add(LSTM(lstm_unit,return_sequences=False))  
model.add(Dense(72, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

LSTM = model.fit(train_x, train_y, epochs=30, batch_size=64, validation_data=(test_x, test_y))
'''  
fig=plt.figure(dpi=128,figsize=(10,3))
plt.plot(frame,x_i,c='blue')#x变换
plt.ylabel('',fontsize=12)
plt.xticks([])  #去掉横坐标值
plt.xlabel("playout time",fontsize=12)
plt.show()
'''
   
plt.plot(LSTM.history['loss'], label='train')
plt.plot(LSTM.history['val_loss'], label='valid')
plt.legend()
plt.show()
