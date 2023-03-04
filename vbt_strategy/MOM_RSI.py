import numpy as np
import pandas as pd
from itertools import product

from .base import BaseStrategy
from numba import njit
import streamlit as st
import vectorbt as vbt

from utils.vbt import plot_CSCV

@njit
def apply_mom_nb(price, window, lower, upper):
    mom_pct = np.full(price.shape, np.nan, dtype=np.float_)
    entry_signal = np.full(price.shape, np.nan, dtype=np.bool_)
    exit_signal = np.full(price.shape, np.nan, dtype=np.bool_)
    for col in range(price.shape[1]):
        for i in range(window, price.shape[0]):
            pct_change = price[i,col]/price[i-window,col] - 1
            mom_pct[i,col] = pct_change
            exit_signal[i,col] = (pct_change < lower)
            entry_signal[i,col] = (pct_change > upper)
            
    return mom_pct, entry_signal, exit_signal

def get_MomInd():
    MomInd = vbt.IndicatorFactory(
        class_name = 'Mom',
        input_names = ['price'],
        param_names = ['window', 'lower', 'upper'],
        output_names = ['mom_pct','entry_signal', 'exit_signal']
    ).from_apply_func(apply_mom_nb)
    
    return MomInd

class MOM_RSIStrategy(BaseStrategy):
    '''Mom_RSI strategy'''
    _name = "MOM_RSI"
    param_dict = {}
    param_def = [
            {
            "name": "window",
            "type": "int",
            "min":  5,
            "max":  30,
            "step": 2   
            },
            {
            "name": "upper",
            "type": "float",
            "min":  0.0,
            "max":  0.1,
            "step": 0.02   
            },
            {
            "name": "lower",
            "type": "float",
            "min":  0.0,
            "max":  0.1,
            "step": 0.02   
            },
    ]

    @vbt.cached_method
    def run(self, calledby='add'):
        #1. initialize the variables
        windows = self.param_dict['window']
        uppers = self.param_dict['upper']
        lowers = self.param_dict['lower']
        close_price = self.stock_dfs[0][1].close
        open_price = self.stock_dfs[0][1].open

        #2. calculate the indicators
        mom_indicator = get_MomInd().run(close_price, window=windows, lower=lowers, upper=uppers,\
            param_product=True)
        mom_entries = mom_indicator.entry_signal
        mom_exits = mom_indicator.exit_signal
        rsi = vbt.RSI.run(
                close_price, 
                window=10,
                short_name='rsi',
                param_product=True)
        rsi_entries = rsi.rsi_crossed_below(20)
        rsi_exits = rsi.rsi_crossed_above(80)

        #3. remove all the name in param_def from param_dict
        for param in self.param_def:
            del self.param_dict[param['name']]

        #4. generate the vbt signal
        if len(windows) > 1:
            rsi_entries_df = mom_entries.copy()
            rsi_exits_df = mom_exits.copy()
            for col in mom_entries.columns:
                rsi_entries_df[col] = rsi_entries
                rsi_exits_df[col] = rsi_exits

            entries = mom_entries | rsi_entries_df
            exits = mom_exits | rsi_exits_df
        else:
            entries = mom_entries | rsi_entries
            exits = mom_exits | rsi_exits
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
                    plot_CSCV(pf, idxmax, self.param_dict['RARM'])
                pf = pf[idxmax]
                
                self.param_dict.update(dict(zip(['window', 'lower', 'upper'], [int(idxmax[0]), round(idxmax[2], 4), round(idxmax[1], 4)])))
        
        self.pf =pf
        return True

