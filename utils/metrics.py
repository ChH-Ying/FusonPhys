import numpy as np
from scipy import signal
from scipy.stats import pearsonr
import math

def get_HR(sig,fs,low=0.85,high=2.8,enable_win=True,harmonic_remove=True):
    if enable_win:
        sig=sig*signal.windows.hann(sig.shape[0],False)
    sig=sig[np.newaxis,:]
    n = 1 if sig.shape[1] == 0 else 2 ** math.ceil(math.log2(sig.shape[1]))
    f,p=signal.periodogram(sig,fs=fs,nfft=n,detrend=False)
    fmask=np.argwhere((f>low)&(f<high))
    masked_f=np.take(f,fmask)
    masked_p=np.take(p,fmask)

    peak_idx, _ = signal.find_peaks(masked_p[:,0])
    sort_idx = np.argsort(masked_p[peak_idx,0])
    sort_idx = sort_idx[::-1]

    peak_idx1 = peak_idx[sort_idx[0]]
    peak_idx2 = peak_idx[sort_idx[1]]

    hr1=np.take(masked_f,peak_idx1,axis=0)[0]*60
    hr2=np.take(masked_f,peak_idx2,axis=0)[0]*60


    if harmonic_remove:
        if np.abs(hr1 - 2 * hr2) < 10:
            hr = hr2
        else:
            hr = hr1
        return hr
    else:
        return hr1

def MAE(list1:np.ndarray,list2:np.ndarray):
    return np.mean(np.abs(list1-list2))

def RMSE(list1:np.ndarray,list2:np.ndarray):
    return np.sqrt(np.mean(np.power(list1-list2,2)))

def PEARSON(list1:np.ndarray,list2:np.ndarray):
    r,_=pearsonr(list1,list2)
    return r

def metrics(list1:np.ndarray,list2:np.ndarray):
    return MAE(list1,list2),RMSE(list1,list2),PEARSON(list1,list2)