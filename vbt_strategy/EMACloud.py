import numpy as np
import pandas as pd
import talib

import streamlit as st
import vectorbt as vbt
import plotly.graph_objects as go
import plotly.express as px
from numba import njit

from .base import BaseStrategy
from utils.vbt import plot_Histogram

def plot_cloud(fig, legend, shortEMA, longEMA, color1, color2):
        longEMA_down = longEMA[longEMA < shortEMA].append(shortEMA[longEMA >= shortEMA]).sort_index()
        longEMA_up = longEMA[longEMA >= shortEMA].append(shortEMA[longEMA < shortEMA]).sort_index()

        fig.add_trace(go.Scatter(x = shortEMA.index, 
                                 y = shortEMA,
                                 line = dict(color = ('rgba(255, 255, 255, 0)')),
                                 showlegend=False,
                                 name = '------'))
        fig.add_trace(go.Scatter(x = longEMA_down.index, 
                                 y = longEMA_down,
                                 line = dict(color = ('rgba(255, 255, 255, 0)')),
                                 fill = "tonexty",
                                 showlegend=False,
                                 fillcolor = color1,
                                 name = legend))                                 
        fig.add_trace(go.Scatter(x = shortEMA.index, 
                                 y = shortEMA,
                                 line = dict(color = ('rgba(255, 255, 255, 0)')),
                                 showlegend=False,
                                 name = '------'))
        fig.add_trace(go.Scatter(x = longEMA_up.index, 
                                 y = longEMA_up,
                                 line = dict(color = ('rgba(255, 255, 255, 0)')),
                                 fill = "tonexty",
                                 fillcolor = color2,
                                 name = legend))    

def plot_EMAClouds(close_price, SL=5, SU=13, ML=100, MU=200):
    # plot the ema clouds of Short and Medium Periods 
    emaSL = talib.EMA(close_price, timeperiod = SL)
    emaSU = talib.EMA(close_price, timeperiod = SU)
    emaML = talib.EMA(close_price, timeperiod = ML)
    emaMU = talib.EMA(close_price, timeperiod = MU)
    fig  = go.Figure()
    plot_cloud(fig, f'EMA {ML}/{MU}', emaML, emaMU, 'rgba(33,150,243,0.3)', 'rgba(255,183,77,0.3)')
    plot_cloud(fig, f'EMA {SL}/{SU}', emaSL, emaSU, 'rgba(76,175,80,0.35)','rgba(244,67,54,0.35)')
    fig.add_trace(go.Scatter(x = close_price.index, y = close_price, line = dict(color = 'blue'), name='Close'))
    fig.update_xaxes(
                # rangeslider_visible = True,
                rangeselector = dict(
                    buttons = list([
                        dict(count = 1, label = '1M', step = 'month', stepmode = 'backward'),
                        dict(count = 6, label = '6M', step = 'month', stepmode = 'backward'),
                        dict(count = 1, label = 'YTD', step = 'year', stepmode = 'todate'),
                        dict(count = 1, label = '1Y', step = 'year', stepmode = 'backward'),
                        dict(step = 'all')]))
                )
    fig.update_layout(legend=dict(orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1,),
                        title_text='EMA Clouds',
                        title_font_size=30,
                        title_x = 0.5,
                        # yaxis_dtick = 20,
                        margin=go.layout.Margin(l=10, r=1, b=10)
                    )
    st.plotly_chart(fig, use_container_width=True)

        # 3. calculate the buy-sell signals
        #  I. Medium Period Strategy
        # entries = np.logical_and(close_price > emaMU, emaML > emaMU)
        # exits = np.logical_and(close_price < emaMU , emaML < emaMU)

@njit(cache=True)
def cal_CloudSignal(close_price, emaSL, emaSU, emaML, emaMU):
    #  II. Short-Medium Period Strategy
        # D - The trend is down  ; U - The trends is up
        signal_MD = emaML < emaMU   # 中线熊
        signal_MU = emaML >= emaMU  # 中线牛

        signal_SD = emaSL < emaSU
        signal_SU = emaSL >= emaSU
        # st.write(signal_MU)
        # 中线、短线都处于上升趋势 ->买入基础
        entries = np.logical_and(signal_MU, signal_SU)
        # 中线、短线都处于下降趋势 ->卖出基础
        exits = np.logical_and(signal_MD, signal_SD)
        # 中线上升趋势，短线下跌趋势，跌入中线ema cloud区域 ->买入
        # entries = np.logical_or(entries, np.logical_and(np.logical_and(signal_MU, signal_SD), np.logical_and(close_price<emaML, close_price>emaMU)))
        entries = np.logical_or(entries, np.logical_and(signal_MU, close_price<emaML))

        # 中线下降趋势，短线上升趋势->买入
        entries = np.logical_or(entries, np.logical_and(signal_MD, signal_SU))

        # 中线上升趋势，短线下降趋势->卖出
        exits = np.logical_or(exits, np.logical_and(signal_MU, signal_SD))
        # 中线上升趋势，短线下降趋势,且close低于中线->卖出
        exits = np.logical_or(exits, np.logical_and(np.logical_and(signal_MU, signal_SD), close_price < emaMU))
        # 中线下降趋势，短线上升趋势，升入中线ema cloud区域 ->卖出
        # exits = np.logical_or(exits, np.logical_and(np.logical_and(signal_MD, signal_SU), np.logical_and(close_price<emaMU, close_price>emaML)))
        exits = np.logical_or(exits, np.logical_and(signal_MD, close_price>emaML))

        #  III. S-Medium Period Strategy
        # 'D' - The trend is down  ; 'U' - The trends is up
        # signal_MD = emaML < emaMU   # 中线熊
        # signal_MU = emaML >= emaMU  # 中线牛
        # entries = np.logical_and(close_price > emaMU, signal_MU)    #中线牛，价高于云->买入
        # exits = np.logical_and(close_price < emaML , signal_MD)     #中线熊，价低于云->卖出
        # entries = np.logical_or(entries, np.logical_and(close_price > emaML, signal_MD))    # 中线熊，价在云中->买入
        # exits = np.logical_or(exits, np.logical_and(close_price < emaMU, signal_MU))    # 中线牛，价低于云->卖出
        return entries, exits

def EMACloudDef(close_price, SL=5, SU=13, ML=100, MU=200):
        #S - Short period   ; M - Medium period
        #L - lower of period  ; U - upper of period
        emaSL = talib.EMA(close_price, timeperiod = SL)
        emaSU = talib.EMA(close_price, timeperiod = SU)
        emaML = talib.EMA(close_price, timeperiod = ML)
        emaMU = talib.EMA(close_price, timeperiod = MU)

        return cal_CloudSignal(close_price, emaSL, emaSU, emaML, emaMU)

EMAClouds = vbt.IndicatorFactory(
    class_name = "EMAClouds",
    input_names = ["close"],
    param_names = ["SL", "SU", "ML", "MU"],
    output_names = ["entries", "exits"]
    ).from_apply_func(
        EMACloudDef,
        SL=5, SU=13, ML=100, MU=200,
        to_2d = False
        )

class EMACloudStrategy(BaseStrategy):
    '''EMACLOUD strategy'''
    _name = "EMACLOUD"
    desc = "“Ideally, 5-12 or 5-13 EMA cloud acts as a fluid trendline for day trades. 8-9 EMA Clouds can be used as pullback Levels –(optional). Additionally, a high level price over or under 34-50 EMA clouds confirms either bullish or bearish bias on the price action for any timeframe” – Ripster <br> \
            reference: <br>\
            &emsp;https://cn.tradingview.com/script/7LPOiiMN-Ripster-EMA-Clouds/"
    param_dict = {}
    param_def = [
            {
            "name": "ShortLower",
            "type": "int",
            "min":  5,
            "max":  15,
            "step": 2   
            },
            {
            "name": "ShortUpper",
            "type": "int",
            "min":  13,
            "max":  30,
            "step": 2   
            },
            {
            "name": "MediumLower",
            "type": "int",
            "min":  50,
            "max":  90,
            "step": 2   
            },
            {
            "name": "MediumUpper",
            "type": "int",
            "min":  100,
            "max":  180,
            "step": 3   
            },
            
        ]

    # @vbt.cached_method
    def run(self, output_bool=False, calledby='add')->bool:
        # 1. initialize the parameters
        close_price = self.stock_dfs[0][1].close
        open_price = self.stock_dfs[0][1].open
        high_price = self.stock_dfs[0][1].high
        low_price = self.stock_dfs[0][1].low
        symbol = self.stock_dfs[0][0]

        SLs = self.param_dict['ShortLower']
        SUs = self.param_dict['ShortUpper']
        MLs = self.param_dict['MediumLower']
        MUs = self.param_dict['MediumUpper']

        ind = EMAClouds.run(close_price, SL=SLs, SU=SUs, ML=MLs, MU=MUs, param_product=True)
        #Don't look into the future
        entries = ind.entries.vbt.signals.fshift()
        exits = ind.exits.vbt.signals.fshift()

        pf = vbt.Portfolio.from_signals(close=close_price, open=open_price, entries=entries, exits=exits, **self.pf_kwargs)
        if self.param_dict['WFO']:
            entries, exits = self.maxSR_WFO(close_price, entries, exits, 'y', 1)
            pf = vbt.Portfolio.from_signals(close=close_price,
                        open = open_price, 
                        entries = entries, 
                        exits = exits, 
                        **self.pf_kwargs)
            self.param_dict = {'WFO': True}
        else:
            pf = vbt.Portfolio.from_signals(close=close_price,
                        open = open_price, 
                        entries = entries, 
                        exits = exits, 
                        **self.pf_kwargs)
            if calledby == 'add':
                SRs = pf.sharpe_ratio()
                idxmax = SRs[SRs != np.inf].idxmax()
                if output_bool:
                    plot_Histogram(close_price, pf, idxmax)
                    plot_EMAClouds(close_price, int(idxmax[0]), int(idxmax[1]), int(idxmax[2]), int(idxmax[3]))
                pf = pf[idxmax]
                self.param_dict = dict(zip(['ShortLower', 'ShortUpper', 'MediumLower' , 'MediumUpper'], [int(idxmax[0]), int(idxmax[1]), int(idxmax[2]), int(idxmax[3])]))        

        self.pf = pf
        return True
