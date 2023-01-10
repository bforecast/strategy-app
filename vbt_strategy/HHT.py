import numpy as np
import pandas as pd

import streamlit as st
import vectorbt as vbt
# import pyhht
from PyEMD import EEMD, CEEMDAN, EMD
from vmdpy import VMD

from plotly.subplots import make_subplots # creating subplots
import plotly.graph_objects as go
from numba import njit
from statsmodels.tsa.stattools import adfuller
from scipy.signal import hilbert
import tftb.processing



from .base import BaseStrategy
from utils.vbt import plot_Histogram

# Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller

#Augmented Dickey-Fuller Test（用于测试稳态）
def adf_test(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    print('ADF Statistic: %f' % dftest[0])
    print('p-value: %f' % dftest[1])
    print('Critical Values:')
    for key, value in dftest[4].items():
        print('\t%s: %.3f' % (key, value))

def find_PeakValley(v, thresh):
    maxthresh = []
    minthresh = []

    entry_signal = np.full(v.shape[0], False, dtype=bool)
    exit_signal = np.full(v.shape[0], False, dtype=bool)
    for x, y in enumerate(v):
        if y > thresh:
            maxthresh.append((x, y))
        elif y < -thresh:
            minthresh.append((x, y))

    for x, y in maxthresh:
        try:
            if (v[x - 1] < y) & (v[x + 1] < y):
                exit_signal[x] = True
        except Exception:
            pass

    for x, y in minthresh:
        try:
            if (v[x - 1] > y) & (v[x + 1] > y):
                entry_signal[x] = True
        except Exception:
            pass
    return entry_signal, exit_signal

def round_int(x):
        if x in [float("-inf"),float("inf")]: return float("nan")
        return int(round(x))

def plotly_EMD(symbol, signal, imfs):
    ls=imfs_max_freq(imfs,1,1000)#算每一段曲线的频率
    cycles = list(map(lambda v: round_int(1/v), ls))
    time_samples = signal.index
    n_imfs = imfs.shape[0]
 
    # plt.figure(num=fignum)
    fig = make_subplots(rows=n_imfs + 1, cols=1, shared_xaxes=True)
 
    fig.add_trace(go.Scatter(x=time_samples, y=signal, name=symbol), row=1, col=1)
    # fig['layout']['yaxis']['title']=symbol
    # Plot the IMFs
    for i in range(n_imfs - 1):
        fig.add_trace(go.Scatter(x=time_samples, y=imfs[i, :], name=f"imf{i}-{cycles[i]}days"),
                    row=i+2, col=1)
        # fig['layout'][f'yaxis{i+2}']['title']=f"cycle:{cycles[i]}"
       
    fig.add_trace(go.Scatter(x=time_samples, y=imfs[-1, :], name="Res."),row = n_imfs + 1, col=1)
    # fig['layout'][f'yaxis{n_imfs+1}']['title']="Res."
    fig.update_layout(title ={'text': f"{symbol}'s EMD分解", 'font_size':30, 'y': 0.98, 'x': 0.5, 'xanchor': 'center'},)
    st.plotly_chart(fig, use_container_width=True)

def plotly_HHTEMD(symbol, signal, imfs):
    time_samples = signal.index
    n_imfs = imfs.shape[0]
 
    # plt.figure(num=fignum)
    fig = make_subplots(rows=n_imfs + 1, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(x=time_samples, y=signal, name=symbol), row=1, col=1)

    # fig['layout']['yaxis']['title']=symbol
    # Plot the IMFs
    for i in range(n_imfs-1):
        # 计算各组分的Hilbert变换
        imfsHT = hilbert(imfs[i])
        # 计算各组分Hilbert变换后的瞬时频率
        instf, timestamps = tftb.processing.inst_freq(imfsHT)
        fig.add_trace(go.Scatter(x=time_samples, y=imfs[i, :], name=f"imf{i}-{round_int(1/np.median(instf))}days"),
                    row=i+2, col=1)

        # fig['layout'][f'yaxis{i+2}']['title']=f"cycle:{cycles[i]}"
       
    fig.add_trace(go.Scatter(x=time_samples, y=imfs[-1, :], name="Res."),row = n_imfs + 1, col=1)
    fig['layout'][f'yaxis{n_imfs+1}']['title']="Res."
    fig.update_layout(title ={'text': f"{symbol}'s EMD分解/周期中位值", 'font_size':30, 'y': 0.98, 'x': 0.5, 'xanchor': 'center'},)
    st.plotly_chart(fig, use_container_width=True)

def plotly_HHTEMDdiff(symbol, signal, imfs1, imfs2):
    time_samples = signal.index
    n_imfs1 = imfs1.shape[0]
    n_imfs2 = imfs1.shape[0]
    n_imfs = min(n_imfs1, n_imfs2)

    # plt.figure(num=fignum)
    fig = make_subplots(rows=n_imfs + 1, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(x=time_samples, y=signal, name=symbol), row=1, col=1)

    # fig['layout']['yaxis']['title']=symbol
    # Plot the IMFs
    for i in range(n_imfs-1):
        # 计算各组分的Hilbert变换
        imfsHT1 = hilbert(imfs1[i])
        imfsHT2 = hilbert(imfs2[i])

        # 计算各组分Hilbert变换后的瞬时频率
        instf1, timestamps1 = tftb.processing.inst_freq(imfsHT1)
        instf2, timestamps2 = tftb.processing.inst_freq(imfsHT2)

        fig.add_trace(go.Scatter(x=time_samples, y=imfs1[i, :], name=f"imf1-{i}-{round_int(1/np.median(instf1))}days"),
                    row=i+2, col=1)
        fig.add_trace(go.Scatter(x=time_samples, y=imfs2[i, :], name=f"imf2-{i}-{round_int(1/np.median(instf2))}days"),
                    row=i+2, col=1)                    

        # fig['layout'][f'yaxis{i+2}']['title']=f"cycle:{cycles[i]}"
       
    fig.add_trace(go.Scatter(x=time_samples, y=imfs1[-1, :], name="Res1."),row = n_imfs + 1, col=1)
    fig.add_trace(go.Scatter(x=time_samples, y=imfs2[-1, :], name="Res2."),row = n_imfs + 1, col=1)

    fig['layout'][f'yaxis{n_imfs+1}']['title']="Res."

    fig.update_layout(title ={'text': f"{symbol}'s EMD分解/周期中位值", 'font_size':30, 'y': 0.98, 'x': 0.5, 'xanchor': 'center'},)
    st.plotly_chart(fig, use_container_width=True)



def imfs_max_freq(imfs,sample_rate,fft_size):
    #计算每一个imfs频谱中最高的那个频率
    n_imfs=imfs.shape[0]
    max_freq=[]
    for i in range(n_imfs-1):
        xs=imfs[i,:][:fft_size]
        xf=np.fft.rfft(xs)/fft_size
        freqs=np.linspace(0,sample_rate/2,fft_size//2+1)
        xfp=20*np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
        max_freq.append(freqs[np.argmax(xfp)])
    return max_freq

# @njit
def imf_max_freq(imf,sample_rate,fft_size):
    #计算imf频谱中最高的那个频率
    xs=imf[:fft_size]
    xf=np.fft.rfft(xs)/fft_size
    freqs=np.linspace(0,sample_rate/2,fft_size//2+1)
    xfp=20*np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    return freqs[np.argmax(xfp)]

def cal_EMD(signal):
    np_signal=np.array(signal)
    emd = EMD(extrema_detection='parabol')
    imfs = emd.emd(np_signal)
    return imfs

def cal_EMDMirror(signal):
    np_signal=np.array(signal)
    np_signal=np.append(np_signal, np_signal[::-1])

    decomposer=EMD(np_signal)
    imfs=decomposer.decompose()
    imfs_half=imfs[..., :int(imfs.shape[1]/2)]
    return imfs_half

def cal_EEMD(signal):
    np_signal=np.array(signal)
    eemd = EEMD(trials=100, noise_width=0.2)
    imfs = eemd.eemd(np_signal)
    return imfs

def cal_CEEMDAN(signal):
    np_signal=np.array(signal)
    imfs = CEEMDAN().ceemdan(np_signal)
    return imfs

def cal_VMD(signal):
    np_signal=np.array(signal)

        #. some sample parameters for VMD  
    alpha = 1000       # moderate bandwidth constraint  
    tau = 0.            # noise-tolerance (no strict fidelity enforcement)  
    K = 10              # 3 modes  
    DC = 0             # no DC part imposed  
    init = 1           # initialize omegas uniformly  
    tol = 1e-7  


    #. Run VMD 
    u, u_hat, omega = VMD(np_signal, alpha, tau, K, DC, init, tol)
    imfs = u.copy()
    for i, imf in enumerate(u):
        imfs[u.shape[0]-i-1] = imf
    return imfs

class HHTStrategy(BaseStrategy):
    '''HHT strategy'''
    _name = "HHT"
    param_def = [
            {
            "name": "n_imf",
            "type": "int",
            "min":  1,
            "max":  6,
            "step": 1   
            },
        ]

    def run(self, calledby='add')->bool:
        #1. initialize the variables
        close_price = self.stock_dfs[0][1].close
        open_price = self.stock_dfs[0][1].open
        symbol = self.stock_dfs[0][0]

        #2. calculate the indicators
        imfs = cal_EMD(close_price)

        if self.output_bool:
            #2.1 plot all imfs
            plotly_HHTEMD(symbol, close_price, imfs)

            #2.2 plot the 1-day difference of all imfs
            imfs2 = cal_EMD(np.append(close_price[:-1], close_price[-2]))
            n_imfs = imfs.shape[0]
            fig = make_subplots(rows=n_imfs, cols=1, shared_xaxes=True)
            for n, imf in enumerate(imfs2):
                if n < len(imfs):
                    fig.add_trace(go.Scatter(x=close_price.index, y=imfs[n]-imf, name=f"imf{n}"),
                                row=n+1, col=1)
                else:
                    st.warning(f'imfs[{n}] is missing.')
            fig.update_layout(title ={'text': f"1天IMFs差异图", 'font_size':30, 'y': 0.98, 'x': 0.5, 'xanchor': 'center'},)
            st.plotly_chart(fig, use_container_width=True)

        n_imf = self.param_dict['n_imf']

        #3. remove all the name in param_def from param_dict
        for param in self.param_def:
            del self.param_dict[param['name']]

        #4. generate the vbt signal
        entries = pd.DataFrame(index=close_price.index)
        exits = pd.DataFrame(index=close_price.index)
        for i in n_imf:
            if i < len(imfs):
                entry_signal, exit_signal = find_PeakValley(imfs[i], 0)
                entries[i] = entry_signal
                exits[i] = exit_signal

        #Don't look into the future
        entries = entries.vbt.signals.fshift()
        exits = exits.vbt.signals.fshift()

        #5. Build portfolios
        if self.param_dict['WFO']!='None':
            if type(n_imf) != int:
                n_imf = n_imf[0]
            n_imf = 2
            num_periods = len(close_price)
            entries = pd.Series(np.full(close_price.shape[0], False, dtype=bool), index=close_price.index)
            exits = pd.Series(np.full(close_price.shape[0], False, dtype=bool), index=close_price.index)
            imf_signal = pd.Series(index=close_price.index)

            window = 12*21  #yearly
            validate_period = 1
            # st.write(entries)
            update_bar = st.progress(0)

            for m in range(0, num_periods-window, validate_period):
                imfs = cal_EMD(close_price[m:m+window])
                if n_imf < len(imfs):
                    imf = imfs[n_imf]
                    if (imf[-2] > imf[-1]) & (imf[-3] < imf[-2]):
                        exits[m+window-1] = True
                    elif (imf[-2] < imf[-1]) & (imf[-3] > imf[-2]):
                        entries[m+window-1] = True
                    imf_signal[m+window-1] = imf[-1]
                else:
                    st.warning(f"imfs[{n_imf}] of {entries.index[m]} is missing.")
                update_bar.progress((m+1) / (num_periods-window))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=imf_signal.index, y=imf_signal, name=f"imf_signal"),)
            st.plotly_chart(fig.add_trace(go.Scatter(x=imf_signal.index[m:], y=imf, name=f"imf{n_imf}")),
                            use_container_width=True)  
            pf = vbt.Portfolio.from_signals(close=close_price,
                        open = open_price, 
                        entries = entries, 
                        exits = exits, 
                        **self.pf_kwargs)
            self.param_dict.update({'n_imf': n_imf})
        else:
            pf = vbt.Portfolio.from_signals(close=close_price, open=open_price, entries=entries, exits=exits, **self.pf_kwargs)
            if calledby == 'add':
                RARMs = eval(f"pf.{self.param_dict['RARM']}()")
                idxmax = RARMs[RARMs != np.inf].idxmax()
                if self.output_bool:
                    plot_Histogram(pf, idxmax, f"Maximize {self.param_dict['RARM']}")
                pf = pf[idxmax]
                self.param_dict.update({'n_imf': idxmax})

        self.pf = pf
        return True
