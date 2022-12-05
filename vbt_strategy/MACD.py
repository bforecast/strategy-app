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
            "min":  5,
            "max":  25,
            "step": 1   
            },
            {
            "name": "slow_window",
            "type": "int",
            "min":  31,
            "max":  51,
            "step": 1   
            },
            {
            "name": "signal_window",
            "type": "int",
            "min":  5,
            "max":  21,
            "step": 1   
            },
        ]

    @vbt.cached_method
    def run(self, output_bool=False, calledby='add'):
        close_price = self.stock_dfs[0][1].close
        open_price = self.stock_dfs[0][1].open

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

        # Long when MACD is above zero AND signal
        entries = macd_ind.macd_above(0) & macd_ind.macd_above(macd_ind.signal)
        # Short when MACD is below zero OR signal
        exits = macd_ind.macd_below(0) | macd_ind.macd_below(macd_ind.signal)

        #Don't look into the future
        entries = entries.vbt.signals.fshift()
        exits = exits.vbt.signals.fshift()

        # Build portfolio
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
                self.param_dict = dict(zip(['fast_window', 'slow_window', 'signal_window'], [int(idxmax[0]), int(idxmax[1]), int(idxmax[2])]))        
        
        self.pf = pf
        return True

   
