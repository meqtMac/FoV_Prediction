# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 19:47:19 2021

@author: admin
"""
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import csv
from matplotlib import pyplot as plt
import glob
import math
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.metrics import mean_squared_error
#from keras.models import Sequential
from keras.models import *
from keras.layers import Dense,Dropout,Multiply
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from keras.layers.core import *
from keras.layers import Bidirectional
import keras.backend as K

lstm_unit=128     #hidden layer units
time_step=5
#batch_size=32    #每一批次训练多少个样例
#input_size=1      #输入层维度
#output_size=1     #输出层维度
#lr=0.0006         #学习率
data_x,data_y=[],[]   #训练集，x存储时间步长的数据，y存储标签
input_dim=1
num_classes = 6
one_hot=[]

def series_to_supervised(data):
    for i in range(len(data)-time_step-1):        
        x=data[i:i+time_step]
        x=np.array(x)
        x=x.reshape(time_step,1)#一维变二维
        y=data[i+time_step:i+time_step+1]
        data_x.append(x.tolist())#训练数据
        data_y.append(y)#标签

#批量读取路径    
csvx_list=glob.glob(r"../../Gaze_txt_files/*/179.*.txt")

for filename in csvx_list:
    x_i=[]
    y_i=[]
    frame=[]
    with open(filename,mode='r', encoding='utf-8') as f:# 打开txt文件，以‘utf-8'编码读
        lines = f.readlines()  # 以行的形式进行读取文件,line存储所有行
        for line in lines:
            a=line.split(",")
            b = float(a[6])  # 这是选取需要读取的位数
            #b = int(6*b)#获取长边块
            c=float(a[1])
            x_i.append(b) # 将其添加在列表之中
            frame.append(c)
    m=[x_i[i] for i in range(0,len(x_i),15)]#500ms取一个视点坐标
    series_to_supervised(m)
    
'''    
for i in range(len(data_y)):
    one_hot_y = np.zeros(num_classes)
    a=data_y[i]
    one_hot_y[a[0]]= 1
    one_hot.append(one_hot_y.tolist())
'''
#data_y=one_hot

'''用22个人做训练，8个人做测试'''
    
#train_x=data_x[:6269]
#test_x=data_x[6270:]
#train_y=data_y[:6269]
#test_y=data_y[6270:]

data_x=np.array(data_x)
data_y=np.array(data_y)

train_x=data_x[:2090]
test_x=data_x[2090:]
train_y=data_y[:2090]
test_y=data_y[2090:]
'''随机选取80%的人做训练，20%的人做预测'''
#train_x, test_x, train_y, test_y = train_test_split(data_x,data_y, test_size=0.2, random_state=33)

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, time_step))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(time_step, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul
'''
def model_attention_applied_before_lstm():
    model = Sequential()
    inputs = Input(shape=(time_step, input_dim,))
    model.add(attention_3d_block(inputs))
    model.add(LSTM(lstm_unit, return_sequences=True,
               input_shape=(time_step, data_dim)))    
    model.add(LSTM(lstm_unit,return_sequences=False)) 
    model.add(Dropout(0.5))#添加后的，提高了一点准确率
    #model.add(Dense(32, activation='relu'))#添加后的，加快了训练速度，准确率基本无变化，准确率波动变大 
    model.add(Dense(12, activation='softmax'))
    return model
'''

def model_attention_applied_after_lstm():
    K.clear_session() #清除之前的模型，省得压满内存
    inputs = Input(shape=(time_step, input_dim,))
    lstm_out = Bidirectional(LSTM(lstm_unit, return_sequences=True))(inputs)
    #lstm_out = LSTM(lstm_unit, return_sequences=True)(lstm_out)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='relu')(attention_mul)#softmax
    model = Model(inputs=inputs, outputs=output)
    return model

SINGLE_ATTENTION_VECTOR = True
model = model_attention_applied_after_lstm()
model.compile(loss='mse',
              optimizer='adam')#categorical_crossentropy,rmsprop,metrics=['accuracy']

model.optimizer.lr.assign(0.001)#更改学习率
LSTM = model.fit(train_x, train_y, epochs=30, batch_size=64, validation_data=(test_x, test_y))

fig=plt.figure(dpi=128,figsize=(10,3))
plt.plot(frame,x_i,c='blue')#x变换
plt.ylabel('',fontsize=12)
plt.xticks([])  #去掉横坐标值
plt.xlabel("playout time",fontsize=12)
plt.savefig("figure/attension_fig01.pdf")
plt.show()

   
plt.plot(LSTM.history['loss'], label='train')
plt.plot(LSTM.history['val_loss'], label='valid')
plt.legend()
plt.savefig("figure/attension_fig02.pdf")
plt.show()


test_predict = model.predict(test_x)
fig=plt.figure(dpi=128,figsize=(15,3))
plt.plot(test_predict, label='train')
plt.plot(test_y, label='valid')
plt.legend()
plt.savefig("figure/attension_fig03.pdf")
plt.show()
