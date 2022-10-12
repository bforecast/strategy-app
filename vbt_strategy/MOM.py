import numpy as np

from numba import njit
import streamlit as st
import vectorbt as vbt

from utils.plot import plot_pf
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

    def run(self, output_bool=False):
        windows = self.param_dict['window']
        uppers = self.param_dict['upper']
        lowers = self.param_dict['lower']
        price = self.stock_dfs[0][1].close
        symbol = self.stock_dfs[0][0]

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
            SRs = pf.sharpe_ratio()
            idxmax = SRs[SRs != np.inf].idxmax()
            pf = pf[idxmax]
            self.param_dict = dict(zip(['window', 'lower', 'upper'], [int(idxmax[0]), round(idxmax[1], 4), round(idxmax[2], 4)]))
        
        self.pf =pf
        return True
