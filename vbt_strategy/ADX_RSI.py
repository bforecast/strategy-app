import numpy as np
import pandas as pd
from itertools import product

from .base import BaseStrategy
from numba import njit
import streamlit as st
import vectorbt as vbt
import talib

from utils.vbt import plot_Histogram

@njit(cache=True)
def AdxRsi_signal_nb(close, adx, rsi_below, rsi_above):
    entry_signal = np.full(close.shape, False, dtype=np.bool_)
    exit_signal = np.full(close.shape, False, dtype=np.bool_)
    for x in range(len(close)):
        if rsi_below[x] and adx[x]:
            entry_signal[x] = True
        elif rsi_above[x] and adx[x]:
            exit_signal[x] = True
    return entry_signal, exit_signal


def AdxRsi_indicators(high, low , close, adx_value=25, rsi_window=14, rsi_value=25):

    rsi = vbt.RSI.run(close, window = rsi_window, short_name="rsi")
    rsi_above = rsi.rsi_above(100-rsi_value).to_numpy()
    rsi_below = rsi.rsi_below(rsi_value).to_numpy()

    adx = talib.ADX(high[:,0], low[:,0], close[:,0], timeperiod= 14)
    adx = adx < adx_value

    return AdxRsi_signal_nb(close, adx, rsi_below, rsi_above)

class ADX_RSIStrategy(BaseStrategy):
    '''ADX_RSI strategy
    Initiate a long order (Buy) whenever the RSI(14) reaches 25 while the ADX is showing a value less than 25.
    Initiate a short order (Sell) whenever the RSI(14) reaches 75 while the ADX is showing a value less than 25.
    https://python.plainenglish.io/creating-a-trading-strategy-based-on-the-adx-indicator-9a310ed4b258
    '''
    _name = "ADX_RSI"
    desc = "**Study on the Strength of the ADX Indicator to Detect Trends** <br>   \
            &emsp;Created by Welles Wilder, the Average Directional Index — ADX is a complex indicator that is used to determine the strength or absence of a trend, be it bullish or bearish. <br>  \
            &emsp;we can use it as it was intended to be used, i.e. as a tool to tell us whether markets are trending or ranging. If we use our usual well-known RSI for a contrarian strategy and filter the signals using the ADX, how would that compare to simply using the RSI without filters? Let us take a look at our updated trading conditions:<br>  \
            &emsp;&emsp;Initiate a long order (Buy) whenever the RSI(14) reaches 25 while the ADX is showing a value less than 25. <br>  \
            &emsp;&emsp;Initiate a short order (Sell) whenever the RSI(14) reaches 75 while the ADX is showing a value less than 25. <br>  \
            reference: <br>\
            &emsp;https://python.plainenglish.io/creating-a-trading-strategy-based-on-the-adx-indicator-9a310ed4b258"
    param_dict = {}
    param_def = [
            {
            "name": "adx_value",
            "type": "int",
            "min":  20,
            "max":  30,
            "step": 2   
            },
            {
            "name": "rsi_window",
            "type": "int",
            "min":  10,
            "max":  20,
            "step": 2   
            },
            {
            "name": "rsi_value",
            "type": "int",
            "min":  20,
            "max":  30,
            "step": 2   
            },            
    ]

    @vbt.cached_method
    def run(self, calledby='add')->bool:
        #1. initialize the variables
        close_price = self.stock_dfs[0][1].close
        open_price =  self.stock_dfs[0][1].open
        high_price = self.stock_dfs[0][1].high
        low_price = self.stock_dfs[0][1].low
        adx_values = self.param_dict['adx_value']
        rsi_windows = self.param_dict['rsi_window']
        rsi_values = self.param_dict['rsi_value']

        #2. calculate the indicators
        ind = vbt.IndicatorFactory(
                class_name = 'AdxRsi',
                input_names = ['high', 'low', 'close'],
                param_names = ['adx_value', 'rsi_window', 'rsi_value'],
                output_names = ['entry_signal', 'exit_signal']
            ).from_apply_func(
                AdxRsi_indicators,
                adx_value=25, 
                rsi_window=14, 
                rsi_value=25
                )
        res = ind.run(
                high_price, low_price, close_price, 
                adx_values,
                rsi_windows,
                rsi_values,
                param_product= True,
                )

        #3. remove all the name in param_def from param_dict
        for param in self.param_def:
            del self.param_dict[param['name']]

        #4. generate the vbt signal
        entries = res.entry_signal
        exits = res.exit_signal

        #Don't look into the future
        entries = entries.vbt.signals.fshift()
        exits = exits.vbt.signals.fshift()

        #5. Build portfolios
        if self.param_dict['WFO']!='None':
            entries, exits = self.maxRARM_WFO(close_price, entries, exits, calledby)
            pf = vbt.Portfolio.from_signals(close=close_price, open=open_price, entries=entries, exits=exits, **self.pf_kwargs)
        else:
            pf = vbt.Portfolio.from_signals(close=close_price, open=open_price, entries=entries, exits=exits, **self.pf_kwargs)
            if calledby == 'add':
                RARMs = eval(f"pf.{self.param_dict['RARM']}()")
                idxmax = RARMs[RARMs != np.inf].idxmax()
                if self.output_bool:
                    plot_Histogram(pf, idxmax, f"Maximize {self.param_dict['RARM']}")
                pf = pf[idxmax]
                
                self.param_dict.update(dict(zip(['adx_value', 'rsi_window', 'rsi_value'], [int(idxmax[0]), int(idxmax[1]),int(idxmax[2])])))
       
        self.pf =pf
        return True

