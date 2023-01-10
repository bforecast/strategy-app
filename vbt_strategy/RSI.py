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
            "step": 2   
            },
            {
            "name": "lower",
            "type": "int",
            "min":  20,
            "max":  31,
            "step": 2   
            },
            {
            "name": "upper",
            "type": "int",
            "min":  70,
            "max":  81,
            "step": 2   
            },
        ]

    @vbt.cached_method
    def run(self, calledby='add')->bool:
        #1. initialize the variables
        close_price = self.stock_dfs[0][1].close
        open_price = self.stock_dfs[0][1].open
        windows = self.param_dict['window']
        uppers = self.param_dict['upper']
        lowers = self.param_dict['lower']

        #2. calculate the indicators
        ind = RSI.run(close_price, window=windows, lower=lowers, upper=uppers, param_product=True)

        #3. remove all the name in param_def from param_dict
        for param in self.param_def:
            del self.param_dict[param['name']]

        #4. generate the vbt signal
        #Don't look into the future
        entries = ind.entries.vbt.signals.fshift()
        exits = ind.exits.vbt.signals.fshift()

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

                self.param_dict.update(dict(zip(['window', 'lower', 'upper'], [int(idxmax[0]), int(idxmax[1]), int(idxmax[2])])))

        self.pf = pf
        return True