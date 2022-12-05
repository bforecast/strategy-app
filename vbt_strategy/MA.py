import numpy as np
import traceback

import streamlit as st
import vectorbt as vbt

from .base import BaseStrategy
from utils.vbt import plot_Histogram

class MAStrategy(BaseStrategy):
    '''MA strategy'''
    _name = "MA"
    param_def = [
            {
            "name": "window",
            "type": "int",
            "min":  2,
            "max":  101,
            "step": 1   
            },
        ]

    @vbt.cached_method
    def run(self, output_bool=False, calledby='add')->bool:
        close_price = self.stock_dfs[0][1].close
        open_price = self.stock_dfs[0][1].open
        if calledby == 'add' or self.param_dict['WFO']:
            window = self.param_dict['window']
            fast_ma, slow_ma = vbt.MA.run_combs(close_price, window=window, r=2, short_names=['fast', 'slow'])
        else:
            fast_windows = self.param_dict['fast_window']
            slow_windows = self.param_dict['slow_window']
            fast_ma = vbt.MA.run(close_price, fast_windows)
            slow_ma = vbt.MA.run(close_price, slow_windows)

        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        #Don't look into the future
        entries = entries.vbt.signals.fshift()
        exits = exits.vbt.signals.fshift()

        if self.param_dict['WFO']:
            entries, exits = self.maxSR_WFO(close_price, entries, exits, 'y', 1)
            pf = vbt.Portfolio.from_signals(close=close_price, open=open_price, entries=entries, exits=exits, **self.pf_kwargs)
            self.param_dict = {'WFO': True}
        else:
            pf = vbt.Portfolio.from_signals(close=close_price, open=open_price, entries=entries, exits=exits, **self.pf_kwargs)
            if calledby == 'add':
                SRs = pf.sharpe_ratio()
                idxmax = SRs[SRs != np.inf].idxmax()
                if output_bool:
                    plot_Histogram(close_price, pf, idxmax)
                pf = pf[idxmax]
                self.param_dict = dict(zip(['fast_window', 'slow_window'], [int(idxmax[0]), int(idxmax[1])]))
        self.pf = pf
        return True
