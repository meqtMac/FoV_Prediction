import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
colors=list(mcolors.TABLEAU_COLORS.keys() )

for m in range(3, 9):
    accu = list()
    filePtr = open('./data./accuracy\\lr_m_tile_l10.txt', 'r')
    for line in filePtr:
        line.strip('\n')
        m_temp = int( line.split(',')[0] )
        if m_temp == m:
            accu.append( float(line.split(',')[1]) )
    filePtr.close()
    del filePtr

    pecNum = list()
    for x in range(1, 100):
        pecNum.append(np.percentile(accu, x))
    txt = 'm={}'
    plt.plot(pecNum, range(1,100), linewidth=1, color=mcolors.TABLEAU_COLORS[colors[m%10]], label=txt.format(m) )

plt.xlabel('Accuracy')
plt.ylabel('CDF')
plt.xticks(ticks=np.arange(0, 1.01, step=0.1) )
plt.yticks(ticks=np.arange(0,101,20 ))
plt.grid()
plt.legend()
plt.savefig('./data./figure\\lr_m_l10.eps')
plt.show()
