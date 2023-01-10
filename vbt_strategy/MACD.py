import numpy as np
import pandas as pd
from itertools import combinations, product

import streamlit as st
import vectorbt as vbt

from .base import BaseStrategy
from utils.vbt import plot_Histogram

class MACDStrategy(BaseStrategy):
    '''MACD strategy'''
    _name = "MACD"
    param_def = [
            {
            "name": "fast_window",
            "type": "int",
            "min":  1,
            "max":  20,
            "step": 2   
            },
            {
            "name": "slow_window",
            "type": "int",
            "min":  20,
            "max":  41,
            "step": 2   
            },
            {
            "name": "signal_window",
            "type": "int",
            "min":  1,
            "max":  20,
            "step": 2   
            },
        ]

    @vbt.cached_method
    def run(self, calledby='add')->bool:
        #1. initialize the variables
        close_price = self.stock_dfs[0][1].close
        open_price = self.stock_dfs[0][1].open

        #2. calculate the indicators
        fast_windows = self.param_dict['fast_window']
        slow_windows = self.param_dict['slow_window']
        signal_windows = self.param_dict['signal_window']

        fast_windows, slow_windows, signal_windows = vbt.utils.params.create_param_combs(
            (product, (product, fast_windows, slow_windows), signal_windows))
        # Run MACD indicator
        macd_ind = vbt.MACD.run(
                close_price,
                fast_window=fast_windows,
                slow_window=slow_windows,
                signal_window=signal_windows,
            )

        #3. remove all the name in param_def from param_dict
        for param in self.param_def:
            del self.param_dict[param['name']]

        #4. generate the vbt signal
        # Long when MACD is above zero AND signal
        entries = macd_ind.macd_above(0) & macd_ind.macd_above(macd_ind.signal)
        # Short when MACD is below zero OR signal
        exits = macd_ind.macd_below(0) | macd_ind.macd_below(macd_ind.signal)

        #Don't look into the future
        entries = entries.vbt.signals.fshift()
        exits = exits.vbt.signals.fshift()

        #5. Build portfolios
        if self.param_dict['WFO']!='None':
            entries, exits = self.maxRARM_WFO(close_price, entries, exits, calledby)
            pf = vbt.Portfolio.from_signals(close=close_price,
                        open = open_price, 
                        entries = entries, 
                        exits = exits, 
                        **self.pf_kwargs)
        else:
            pf = vbt.Portfolio.from_signals(close=close_price,
                        open = open_price, 
                        entries = entries, 
                        exits = exits, 
                        **self.pf_kwargs)
            if calledby == 'add':
                RARMs = eval(f"pf.{self.param_dict['RARM']}()")
                idxmax = RARMs[RARMs != np.inf].idxmax()
                if self.output_bool:
                    plot_Histogram(pf, idxmax, f"Maximize {self.param_dict['RARM']}")
                pf = pf[idxmax]
                self.param_dict.update(dict(zip(['fast_window', 'slow_window', 'signal_window'], [int(idxmax[0]), int(idxmax[1]), int(idxmax[2])])))
                
        self.pf = pf
        return True

   
