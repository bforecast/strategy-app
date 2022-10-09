import numpy as np
import pandas as pd
from itertools import product

import streamlit as st
import vectorbt as vbt

from .base import BaseStrategy


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


    def run(self, output_bool=False):
        close_price = self.ohlcv_list[0][1].close
        open_price = self.ohlcv_list[0][1].open
        
        windows = self.param_dict['window']
        upper_ths = self.param_dict['upper']
        lower_ths = self.param_dict['lower']
        lower_ths_prod, upper_ths_prod = zip(*product(lower_ths, upper_ths))

        rsi = vbt.RSI.run(
                open_price, 
                window=windows,
                short_name='rsi',
                param_product=True)


        entries = rsi.rsi_crossed_below(lower_ths_prod, level_name='lower')
        exits = rsi.rsi_crossed_above(upper_ths_prod, level_name='upper')

        pf_kwargs = dict(fees=0.001, freq='1D')
        pf = vbt.Portfolio.from_signals(
            close=close_price, 
            entries=entries, 
            exits=exits,
            size=100,
            size_type='value',
            init_cash='auto',
            **pf_kwargs
            )

        if len(windows) > 1:
            if output_bool:
                # Draw all window combinations as a 3D volume
                st.plotly_chart(
                    pf.total_return().vbt.volume(
                            x_level='upper',
                            y_level='lower',
                            z_level='rsi_window',

                            trace_kwargs=dict(
                                colorbar=dict(
                                    title='Total return', 
                                    tickformat='%'
                                )
                            )
                        )
                    )
                idxmax = (pf.total_return().idxmax())
                # st.write(idxmax)

            idxmax = (pf.sharpe_ratio().idxmax())
            pf = pf[idxmax]
            self.param_dict = dict(zip(['window', 'lower', 'upper'], [int(idxmax[2]), int(idxmax[0]), int(idxmax[1])]))        
        self.pf = pf
   
