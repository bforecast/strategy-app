import numpy as np
import pandas as pd
from itertools import product
import talib

import streamlit as st
import vectorbt as vbt

from .base import BaseStrategy
from utils.vbt import plot_Histogram

def RSIDef(close, window=14, lower=20, upper=80):
    rsi = talib.RSI(close, window)
    entries =  rsi > upper
    exits = rsi < lower
    return entries, exits

RSI = vbt.IndicatorFactory(
    class_name = "RSI",
    input_names = ["close"],
    param_names = ["window", "lower", "upper"],
    output_names = ["entries", "exits"]
    ).from_apply_func(
        RSIDef,
        window = 14,
        lower = 20,
        upper = 80,
        to_2d = False
        )

class RSIStrategy(BaseStrategy):
    '''RSI strategy'''
    _name = "RSI"
    param_def = [
            {
            "name": "window",
            "type": "int",
            "min":  8,
            "max":  21,
            "step": 1   
            },
            {
            "name": "lower",
            "type": "int",
            "min":  20,
            "max":  31,
            "step": 1   
            },
            {
            "name": "upper",
            "type": "int",
            "min":  70,
            "max":  81,
            "step": 1   
            },
        ]

    @vbt.cached_method
    def run(self, output_bool=False, calledby='add'):
        close_price = self.stock_dfs[0][1].close
        open_price = self.stock_dfs[0][1].open
        windows = self.param_dict['window']
        uppers = self.param_dict['upper']
        lowers = self.param_dict['lower']

        ind = RSI.run(close_price, window=windows, lower=lowers, upper=uppers, param_product=True)
        #Don't look into the future
        entries = ind.entries.vbt.signals.fshift()
        exits = ind.exits.vbt.signals.fshift()

        if self.param_dict['WFO']:
            exits.columns = entries.columns
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
                self.param_dict = dict(zip(['window', 'lower', 'upper'], [int(idxmax[0]), int(idxmax[1]), int(idxmax[2])]))        
        self.pf = pf
        return True