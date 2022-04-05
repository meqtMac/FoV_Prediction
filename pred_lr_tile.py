## Prediction Using LR
## import
import os
from os.path import join
from matplotlib import pyplot as plt
import numpy as np
from scipy.sparse.construct import random
import func
from sklearn.metrics import r2_score

index_dict = {'1':'xtrace', '2':'ytrace' }
index = 1
playNum = len( os.listdir(func.path) )
accuracy = list()

print( 'hello' )

## Prediction
for m in range(3, 9):
    accu = list()
    print(m)
    for player in os.listdir(func.path)[int(playNum*func.train_rate):]:
        print(player)
        dirPath = func.path+'./'+player
        for video in os.listdir(dirPath):
            filePtr = open( join(dirPath, video) )
            test_window = func.load_test_window( filePtr, index, m+1 )
            filePtr.close()
            del filePtr
            pred_lr = func.prediction_lr(test_window)
            accu.append( func.accu_tile(pred_lr, test_window[:,-1] ) )
    accuracy.append(accu)

## OutPut with accuracy CDF measured by r2_score
import matplotlib.colors as mcolors
colors=list(mcolors.TABLEAU_COLORS.keys() )

filePtr = open('./data./accuracy\\lr_m_tile_l15.txt', 'w')
for m in range(3, 9):
    accu = accuracy[m-3]
    accu=np.array(accu)
    pecNum = list()
    for a in range( len(accu) ):
        txt='{},{}\n'
        filePtr.write(txt.format(m, accu[a]))
    for x in range(1, 100):
        pecNum.append( np.percentile(accu,x))
    txt='m={}'
    plt.plot(pecNum, range(1,100), linewidth=1, color=mcolors.TABLEAU_COLORS[colors[m%10]], label=txt.format(m) )
## append new data
filePtr.close()
plt.xlabel('Accuracy')
plt.ylabel('CDF')
plt.grid()
plt.legend()
txt = 'Accuracy CDF when intervel = {}'
plt.title(txt.format(15))
plt.savefig('./data./figure\\lr_m_l15.eps')
plt.show()
