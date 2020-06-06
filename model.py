import glob
import pandas as pd
import numpy as np
import wfdb

paths = glob.glob("data/*.dat")
paths = [path[:-4] for path in paths]
paths.sort()

def segmentation(records,typeBeats):
    Normal = []
    for e in records:
        signals, fields = wfdb.rdsamp(e, channels = [0]) 

        # plot result 
        wfdb.plot_items(signal=signals, fs=fields['fs'], title='')
        
        ann = wfdb.rdann(e, 'q1c')
        good = [typeBeats]
        ids = np.in1d(ann.symbol, good)
        imp_beats = ann.sample[ids]
        beats = (ann.sample)
        for i in imp_beats:
            beats = list(beats)
            j = beats.index(i)
            if(j!=0 and j!=(len(beats)-1)):
                x = beats[j-1]
                y = beats[j+1]
                diff1 = abs(x - beats[j])//2
                diff2 = abs(y - beats[j])//2
                Normal.append(signals[beats[j] - diff1: beats[j] + diff2, 0])
    return Normal

def smooth(x,window_len=11,window='hanning'):
    if x.ndim != 1:
        raise ValueError #"smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError #"Input vector needs to be bigger than window size."

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError # "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    print(x)
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    # print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

N=pd.DataFrame(segmentation(paths,'N'))
N=N.iloc[:,:270]
N=N.dropna()
N=np.array(N)

total = N.shape[0]

# new array M for storing results
M = [None] * total

for i in range(total):
    M[i]=smooth(N[0,:])
