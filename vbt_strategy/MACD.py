import numpy as np
import pandas as pd
from itertools import combinations, product

import streamlit as st
import vectorbt as vbt

from .base import BaseStrategy


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


    def run(self, output_bool=False):
        price = self.stock_dfs[0][1].close
        
        fast_windows = self.param_dict['fast_window']
        slow_windows = self.param_dict['slow_window']
        signal_windows = self.param_dict['signal_window']

        fast_windows, slow_windows, signal_windows = vbt.utils.params.create_param_combs(
            (product, (product, fast_windows, slow_windows), signal_windows))
        # Run MACD indicator
        macd_ind = vbt.MACD.run(
                price,
                fast_window=fast_windows,
                slow_window=slow_windows,
                signal_window=signal_windows,
            )

        # Long when MACD is above zero AND signal
        entries = macd_ind.macd_above(0) & macd_ind.macd_above(macd_ind.signal)

        # Short when MACD is below zero OR signal
        exits = macd_ind.macd_below(0) | macd_ind.macd_below(macd_ind.signal)

        # Build portfolio
        pf = vbt.Portfolio.from_signals(
            price, entries, exits, fees=0.001, freq='1D')

        if len(fast_windows) > 1:
            if output_bool:
                # Draw all window combinations as a 3D volume
                st.plotly_chart(
                    pf.total_return().vbt.volume(
                    x_level='macd_fast_window',
                    y_level='macd_slow_window',
                    z_level='macd_signal_window',
                    trace_kwargs=dict(
                            colorbar=dict(
                                title='Total return', 
                                tickformat='%'
                            )
                        )
                    )
                )

            SRs = pf.sharpe_ratio()
            idxmax = SRs[SRs != np.inf].idxmax()
            pf = pf[idxmax]
            self.param_dict = dict(zip(['fast_window', 'slow_window', 'signal_window'], [int(idxmax[0]), int(idxmax[1]), int(idxmax[2])]))        
        self.pf = pf
        return True

   
