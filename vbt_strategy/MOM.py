import numpy as np

from numba import njit
import streamlit as st
import vectorbt as vbt

from .base import BaseStrategy
from utils.vbt import plot_Histogram


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

class MOMStrategy(BaseStrategy):
    '''Mom strategy'''
    _name = "MOM"
    param_dict = {}
    param_def = [
            {
            "name": "window",
            "type": "int",
            "min":  5,
            "max":  30,
            "step": 1   
            },
            {
            "name": "upper",
            "type": "float",
            "min":  0.0,
            "max":  0.1,
            "step": 0.01   
            },
            {
            "name": "lower",
            "type": "float",
            "min":  0.0,
            "max":  0.1,
            "step": 0.01   
            },
    ]

    @vbt.cached_method
    def run(self, output_bool=False, calledby='add'):
        windows = self.param_dict['window']
        uppers = self.param_dict['upper']
        lowers = self.param_dict['lower']
        close_price = self.stock_dfs[0][1].close
        open_price = self.stock_dfs[0][1].open

        mom_indicator = get_MomInd().run(close_price, window=windows, lower=lowers, upper=uppers,\
            param_product=True)
          
        entries = mom_indicator.entry_signal.vbt.signals.fshift()
        exits = mom_indicator.exit_signal.vbt.signals.fshift()

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
                self.param_dict = dict(zip(['window', 'lower', 'upper'], [int(idxmax[0]), round(idxmax[1], 4), round(idxmax[2], 4)]))

        self.pf =pf
        return True