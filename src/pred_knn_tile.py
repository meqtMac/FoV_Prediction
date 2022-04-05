import os
from os.path import join
from matplotlib import pyplot as plt
import numpy as np
import func

index_dict = {'1':'xtrace', '2':'ytrace' }
index = 1
playNum = len( os.listdir(func.path) )
accuracy = list()

print( 'hello' )

for k in range(3, 9):
    accu = list()
    print(k)
    for player in os.listdir(func.path)[int(playNum*func.train_rate):]:
        dirPath = func.path+'./'+player
        print(player)
        for video in os.listdir(dirPath):
            filePtr = open( join(dirPath, video) )
            test_window = func.load_test_window( filePtr, index, func.sw_width)
            train_window = func.load_train_window(func.path, func.train_rate, int((video.split('.'))[0]), index,func.sw_width )
            filePtr.close()
            del filePtr
            pred_knn = func.prediction_knn(test_window, train_window, k)
            accu.append( func.accu_tile(pred_knn, test_window[:,-1] ) )
    accuracy.append(accu)

import matplotlib.colors as mcolors
colors=list(mcolors.TABLEAU_COLORS.keys())
filePtr = open( './data./accuracy\\knn_k_tile_i15.txt','w')

for k in range(3, 9):
    accu = accuracy[k-3]
    accu=np.array(accu)
    for a in range( len(accu) ):
        txt = '{},{}\n'
        filePtr.write( txt.format(k, accu[a] ) )
    pecNum = list()
    for x in range(1, 100):
        pecNum.append( np.percentile(accu,x))
    
    txt='k={}'
    plt.plot(pecNum, range(1,100), linewidth=1, color=mcolors.TABLEAU_COLORS[colors[k%10]], label=txt.format(k) )
#append new data
filePtr.close()
plt.xticks( ticks=np.arange(0, 1.01, step=0.1) )
plt.yticks( ticks=np.arange(0, 101, step=20) )
plt.xlabel('Accuracy')
plt.ylabel('CDF')
plt.grid()
plt.legend()
plt.savefig('./data./figure\\knn_k_i15.eps')
plt.show()
