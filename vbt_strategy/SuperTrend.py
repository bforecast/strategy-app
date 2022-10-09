import numpy as np

from numba import njit
import streamlit as st
import vectorbt as vbt

import talib
from .base import BaseStrategy


@njit
def get_final_bands_nb(close, upper, lower):
    trend = np.full(close.shape, np.nan)
    dir_ = np.full(close.shape, 1)
    long = np.full(close.shape, np.nan)
    short = np.full(close.shape, np.nan)

    for i in range(1, close.shape[0]):
        if close[i] > upper[i - 1]:
            dir_[i] = 1
        elif close[i] < lower[i - 1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i - 1]
            if dir_[i] > 0 and lower[i] < lower[i - 1]:
                lower[i] = lower[i - 1]
            if dir_[i] < 0 and upper[i] > upper[i - 1]:
                upper[i] = upper[i - 1]

        if dir_[i] > 0:
            trend[i] = long[i] = lower[i]
        else:
            trend[i] = short[i] = upper[i]
            
    return trend, dir_, long, short

@njit
def get_basic_bands(med_price, atr, multiplier):
    matr = multiplier * atr
    upper = med_price + matr
    lower = med_price - matr
    return upper, lower

def faster_supertrend_talib(high, low, close, window, multiplier):
    avg_price = np.full(close.shape, np.nan)
    atr = np.full(close.shape, np.nan)
    upper = np.full(close.shape, np.nan)
    lower = np.full(close.shape, np.nan)
    for col in range(close.shape[1]):
        avg_price[:,col] = talib.MEDPRICE(high[:,col], low[:,col])
        atr[:,col] = talib.ATR(high[:,col], low[:,col], close[:,col], window)
        upper[:,col], lower[:,col] = get_basic_bands(avg_price[:,col], atr[:,col], multiplier)
    return get_final_bands_nb(close, upper, lower)       

class SuperTrendStrategy(BaseStrategy):
    '''SuperTrend strategy'''
    _name = "SuperTrend"
    param_dict = {}
    param_def = [
            {
            "name": "window",
            "type": "int",
            "min":  4,
            "max":  20,
            "step": 1   
            },
            {
            "name": "multiplier",
            "type": "float",
            "min":  2.0,
            "max":  4.1,
            "step": 0.1   
            },
    ]

    def run(self, output_bool=False):
        windows = self.param_dict['window']
        multipliers = self.param_dict['multiplier']
        high = self.ohlcv_list[0][1].high
        low = self.ohlcv_list[0][1].low
        close = self.ohlcv_list[0][1].close
        symbol = self.ohlcv_list[0][0]

        SuperTrend = vbt.IndicatorFactory(
            class_name='SuperTrend',
            input_names=['high', 'low', 'close'],
            param_names=['window', 'multiplier'],
            output_names=['supert', 'superd', 'superl', 'supers']
        ).from_apply_func(
            faster_supertrend_talib
        )

        st_indicator = SuperTrend.run(
                high, low, close, 
                window = windows, 
                multiplier = multipliers,
                param_product=True,
            )
        entries = (~st_indicator.superl.isnull()).vbt.signals.fshift()
        exits = (~st_indicator.supers.isnull()).vbt.signals.fshift()
        pf = vbt.Portfolio.from_signals(
                    close=close, 
                    entries=entries, 
                    exits=exits, 
                    fees=0.001, 
                    freq='1d'
                )
        
        if output_bool:
            # Draw all window combinations as a 2D volume
            fig = pf.total_return().vbt.heatmap(
                        x_level='supertrend_window', 
                        y_level='supertrend_multiplier',
                    )
            st.plotly_chart(fig)

        if len(windows) > 1:
            idxmax = (pf.sharpe_ratio().idxmax())
            pf = pf[idxmax]
            self.param_dict = dict(zip(['window', 'multiplier'], [int(idxmax[0]), round(idxmax[1], 1)]))
        
        self.pf =pf
