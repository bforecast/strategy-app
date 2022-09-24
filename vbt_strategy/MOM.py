import numpy as np
import pandas as pd
from datetime import datetime

from numba import njit
import streamlit as st
import vectorbt as vbt

from utils.vbt_nb import plot_pf
from .base import BaseStrategy


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

    def run(self, output_bool=False):
        windows = self.param_dict['window']
        uppers = self.param_dict['upper']
        lowers = self.param_dict['lower']
        price = self.ohlcv_list[0][1].close
        symbol = self.ohlcv_list[0][0]

        if output_bool:
            st.write("Calculate stock " + symbol)

        mom_indicator = get_MomInd().run(price, window=windows, lower=lowers, upper=uppers,\
            param_product=True)
        entries = mom_indicator.entry_signal
        exits = mom_indicator.exit_signal
        pf = vbt.Portfolio.from_signals(price, entries, exits, fees=0.002, freq='1D')
        
        if output_bool:
            # Draw all window combinations as a 3D volume
            fig = pf.total_return().vbt.volume(
                x_level='mom_upper',
                y_level='mom_lower',
                z_level='mom_window',

                trace_kwargs=dict(
                    colorbar=dict(
                        title='Total return', 
                        tickformat='%'
                    )
                )
            )
            st.plotly_chart(fig)

        if len(windows) > 1:
            idxmax = (pf.sharpe_ratio().idxmax())
            pf = pf[idxmax]
            self.param_dict = dict(zip(['window', 'lower', 'upper'], [int(idxmax[0]), round(idxmax[1], 4), round(idxmax[2], 4)]))
        return pf
                


    def maxSR(self, output_bool=False):
        self.param_dict = {
            "window":   np.arange(5, 30),
            'upper':    np.arange(0, 0.1, 0.01),
            'lower':    np.arange(0, 0.1, 0.01)
        }

        pf = self.run(output_bool)
        if output_bool:
            plot_pf(pf)
       
        return self.param_dict, pf
    
