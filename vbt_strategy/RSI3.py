import numpy as np
import pandas as pd
import talib
from itertools import product

import streamlit as st
import vectorbt as vbt

from .base import BaseStrategy
from utils.vbt import plot_Histogram

def threeRSIDef(close, window1=14, window2=50, window3=100):
    rsi1 = talib.RSI(close, window1)
    rsi2 = talib.RSI(close, window2)
    rsi3 = talib.RSI(close, window3)
    entries = np.where((rsi1 > rsi2) & (rsi2 > rsi3), True, False)
    exits = np.where((rsi1 < rsi2) & (rsi2 < rsi3), True, False)
    return entries, exits

threeRSI = vbt.IndicatorFactory(
    class_name = "threeRSI",
    short_name = "RSI3",
    input_names = ["close"],
    param_names = ["window1", "window2", "window3"],
    output_names = ["entries", "exits"]
    ).from_apply_func(
        threeRSIDef,
        window1 = 14,
        window2 = 30,
        window3 = 70,
        to_2d = False
        )


class RSI3Strategy(BaseStrategy):
    '''RSI3 strategy'''
    _name = "RSI3"
    param_def = [
            {
            "name": "window1",
            "type": "int",
            "min":  2,
            "max":  20,
            "step": 2   
            },
            {
            "name": "window2",
            "type": "int",
            "min":  20,
            "max":  80,
            "step": 4   
            },
            {
            "name": "window3",
            "type": "int",
            "min":  80,
            "max":  250,
            "step": 8   
            },
        ]

    @vbt.cached_method
    def run(self, output_bool=False, calledby='add'):
        close_price = self.stock_dfs[0][1].close
        open_price = self.stock_dfs[0][1].open
        
        windows1 = self.param_dict['window1']
        windows2 = self.param_dict['window2']
        windows3 = self.param_dict['window3']

        ind = threeRSI.run(close_price, window1=windows1, window2=windows2, window3=windows3,\
            param_product=True)
        #Don't look into the future
        entries = ind.entries.vbt.signals.fshift()
        exits = ind.exits.vbt.signals.fshift()

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
                pf = pf[idxmax]
                self.param_dict = dict(zip(['window1', 'window2', 'window3'], [int(idxmax[0]), int(idxmax[1]), int(idxmax[2])]))        
        
        self.pf = pf
        return True
   
