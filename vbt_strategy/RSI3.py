import numpy as np
import pandas as pd
import talib
from itertools import product

import streamlit as st
import vectorbt as vbt

from .base import BaseStrategy

def threeRSIDef(close, window1=14, window2=50, window3=100, case=0):
    rsi1 = talib.RSI(close, window1)
    rsi2 = talib.RSI(close, window2)
    rsi3 = talib.RSI(close, window3)

    #there must be a better way of doing this
    if case == 0:
        
        entries = np.where((rsi1 > rsi2) & (rsi2 > rsi3), True, False)
        exits = np.where((rsi1 < rsi2) & (rsi2 < rsi3), True, False)

        
    elif case == 1:
        
        trend = np.where((rsi1 > rsi2) & (rsi1 > rsi3) & (rsi2 < rsi3), 1, -1)
        
    elif case == 2:
        
        trend = np.where((rsi1 < rsi2) & (rsi1 > rsi3) & (rsi2 > rsi3), 1, -1)
        
    elif case == 3:
        
        trend = np.where((rsi1 > rsi2) & (rsi1 < rsi3) & (rsi2 < rsi3), 1, -1)
        
    elif case == 4:
        
        trend = np.where((rsi1 < rsi2) & (rsi1 < rsi3) & (rsi2 > rsi3), 1, -1)
        
    elif case == 5:
        
        trend = np.where((rsi1 < rsi2) & (rsi1 < rsi3) & (rsi2 < rsi3), 1, -1)

    return entries, exits

threeRSI = vbt.IndicatorFactory(
    class_name = "threeRSI",
    short_name = "RSI3",
    input_names = ["close"],
    param_names = ["window1", "window2", "window3", "case"],
    output_names = ["entries", "exits"]
    ).from_apply_func(
        threeRSIDef,
        window1 = 14,
        window2 = 30,
        window3 = 70,
        case = 0,
        to_2d = False
        )


class RSI3Strategy(BaseStrategy):
    '''RSI3 strategy'''
    _name = "RSI3"
    param_def = [
            {
            "name": "window1",
            "type": "int",
            "min":  2,
            "max":  20,
            "step": 2   
            },
            {
            "name": "window2",
            "type": "int",
            "min":  20,
            "max":  80,
            "step": 4   
            },
            {
            "name": "window3",
            "type": "int",
            "min":  80,
            "max":  250,
            "step": 8   
            },
        ]


    def run(self, output_bool=False):
        close_price = self.stock_dfs[0][1].close
        open_price = self.stock_dfs[0][1].open
        
        windows1 = self.param_dict['window1']
        windows2 = self.param_dict['window2']
        windows3 = self.param_dict['window3']

        ind = threeRSI.run(close_price, window1=windows1, window2=windows2, window3=windows3,\
            param_product=True)

        pf_kwargs = dict(fees=0.001, freq='1D')
        pf = vbt.Portfolio.from_signals(
            close=close_price, 
            open=open_price, 
            entries=ind.entries, 
            exits=ind.exits,
            size=100,
            size_type='value',
            init_cash='auto',
            **pf_kwargs
            )

        if len(pf.total_return()) > 0:
            if output_bool:
                # Draw all window combinations as a 3D volume
                st.plotly_chart(
                    pf.total_return().vbt.volume(
                            x_level='RSI3_window1',
                            y_level='RSI3_window2',
                            z_level='RSI3_window3',

                            trace_kwargs=dict(
                                colorbar=dict(
                                    title='Total return', 
                                    tickformat='%'
                                )
                            )
                        )
                    )
                idxmax = (pf.total_return().idxmax())

            SRs = pf.sharpe_ratio()
            idxmax = SRs[SRs != np.inf].idxmax()
            pf = pf[idxmax]
            self.param_dict = dict(zip(['window1', 'window2', 'window3'], [int(idxmax[0]), int(idxmax[1]), int(idxmax[2])]))        
        self.pf = pf
        return True
   
