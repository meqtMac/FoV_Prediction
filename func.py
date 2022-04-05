import os
from os.path import join
import numpy as np
from sklearn import neighbors
from sklearn import linear_model
import matplotlib.pyplot as plt

#basic data
path = str( './data./FoVPreddemo_JY./Gaze_txt_files')
sw_width = int(6)
train_rate = float(0.8)
interval = int(15)
#number of neighbor

#load data from a list to a ndarray
def load_data(source, index, varType):
    data = source[:,index]
    data = data.astype(varType)
    data = data.reshape( len(data), 1 )
    return data 

#load the frame from the trace information and reshape 
def load_frame(source, index, varType):
    frame = source[:,index]
    frame = frame.astype(varType)
    frame = frame.reshape( len(frame), 1)
    return frame

#load trace info without reshape
def load_trace(source, index, varType):
    trace = source[:, index]
    trace = trace.astype(varType)
    return trace

#load the trace inside the file
def load_file_trace(filePtr, index=1, interval=5):
    trace = []
    for line in filePtr:
        line = line.replace('\n','')
        intel = list()
        intel.append( line.split(',')[1] )
        intel.append( line.split(',')[6] )
        intel.append( line.split(',')[7] )
        trace.append( intel)
    del line
    del intel
    trace = np.array( trace[::int(interval)] )
    otrace = load_trace( trace, index, float)
    del trace
    return otrace

# slice data in a video into windows
def sliding_window(source, sw_width):
    windows=[]
    for i in range( len(source)-sw_width+1 ):
        window=[]
        window = source[i:i+sw_width]
        windows.append(window)
    return np.array( windows )

#depict the result of prediction
def plot(data, pred, player=1, video=1, algo='reg'):
    plt.plot( range( len(data) ), data, color='r', label='data' )
    plt.plot( range( len(pred) ), pred, color='c', label='predice')
    txt = "Prediction using: {}, the result of player: {}, video: {}"
    plt.title(txt.format(algo, player, video) )
    plt.grid()
    plt.legend()
    plt.show()

#depict both result
def plot_comb(data, pred_knn, pred_lr,player, video):
    plt.plot( range(len(data)), data, color='r', label='data' )
    plt.plot( range(len(pred_knn)), pred_knn, color='m', label='predict_knn')
    plt.plot( range(len(pred_lr)), pred_lr, color='c', label='predic_lr')
    txt='Prediction result of player: {}, video: {}'
    plt.title( txt.format(player, video) )
    plt.grid()
    plt.legend()
    plt.show()

# predic with the test dataset and output the prediction result
def prediction_knn( test_window, train_window, k):
    knn = neighbors.KNeighborsRegressor(n_neighbors=k, weights='distance', algorithm='kd_tree')
    X = train_window[:,:-1]
    y = train_window[:,-1:]
    y = y.reshape( len(y) )
    T=  test_window[:,:-1]
    pred = knn.fit(X, y).predict(T)
    return pred

# predict with lr
def prediction_lr( test_window ):
    lr = linear_model.LinearRegression( positive=True )
    pred = np.array( list() )
    for frame in range(len(test_window) ):
            y = test_window[frame]
            x = np.array(  range( len(y) )  )
            x = x.reshape( len(x), 1)
            X_train = x[:-1]
            y_train = y[:-1]
            X_test = x[-1:]
            lr.fit( X_train, y_train)
            pred = np.concatenate(  ( pred, lr.predict(X_test) )  )
            for x in range( len(pred) ):
                if pred[x] > 1:
                    pred[x] = 1
    return pred

#output an float of accuracy with tile metric
def accu_tile(pred, test):
    acc = int(0)
    for x in range(len(pred)):
        if int(pred[x]*12)==int(test[x]*12):
            acc += 1
    return float(acc/len(pred))

#load train datasets
def load_train_window(path, train_rate, video, index, sw_width ):
    xtrace_train_window = np.array( list() )

    for player in range(0, int(len(os.listdir(path))*train_rate) ):
        dirPath = path+'./'+os.listdir(path)[player]
        
        for vIndex in range(0, int(  len( os.listdir(dirPath) )  ) ):
            fileName= os.listdir(dirPath)[vIndex]
            
            if int(fileName.split('.')[0]) == video:
                filePtr = open( join(dirPath, fileName) )
                xtrace = load_file_trace(filePtr, index, interval)

                if bool( len(xtrace_train_window) ):
                    xtrace_train_window = np.concatenate(  ( xtrace_train_window, sliding_window(xtrace, sw_width) )  )
                else:
                    xtrace_train_window = sliding_window(xtrace, sw_width)
                
                del xtrace
                filePtr.close()
                del filePtr
    return xtrace_train_window

#load the the summary of train datasets
def load_train_sum(path, train_rate, index, sw_width):
    xtrace_train_window = np.array(list())
    for player in range(  0, int( len(os.listdir(path))*train_rate )  ):
        dirPath = path+'./'+os.listdir(path)[player]
        for fileName in os.listdir(dirPath):
            filePtr=open( join(dirPath, fileName) )
            xtrace = load_file_trace(filePtr, index, interval)
            if bool( len(xtrace_train_window) ):
                    xtrace_train_window = np.concatenate(  ( xtrace_train_window, sliding_window(xtrace, sw_width) )  )
            else:
                xtrace_train_window = sliding_window(xtrace, sw_width)
            
            del xtrace
            filePtr.close()
            del filePtr
    return xtrace_train_window

#load test datasets
def load_test_window(filePtr, index, sw_width):
    xtrace = load_file_trace(filePtr, index, interval) 
    xtrace_test_window = sliding_window(xtrace, sw_width)
    del xtrace
    return xtrace_test_window
