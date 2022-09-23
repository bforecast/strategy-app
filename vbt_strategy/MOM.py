import numpy as np
import pandas as pd
from datetime import datetime

from numba import njit
import streamlit as st
import vectorbt as vbt

from utils.vbt_nb import plot_pf


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

def MOM_MaxSR(symbol:str, price):        
        st.subheader('Search The Parameters for Max Sharpe Ratio')
        # stock_df.index = pd.to_datetime(stock_df['date'])
        # prices = stock_df['close']
        windows = np.arange(5, 30)
        uppers = np.arange(0, 0.1, 0.01)
        lowers = np.arange(0, 0.1, 0.01)
        mom_indicator = get_MomInd().run(price, window=windows, lower=lowers, upper=uppers,\
            param_product=True)
        entries = mom_indicator.entry_signal
        exits = mom_indicator.exit_signal
        pf = vbt.Portfolio.from_signals(price, entries, exits, fees=0.002, freq='1D')
        idxmax = (pf.total_return().idxmax())
        col1, col2 = st.columns(2)
        with col1:
            st.write('Max Return: (Period, Lower, Upper)')
        with col2:
            st.write(idxmax)
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
        idxmax = (pf.sharpe_ratio().idxmax())
        return_pf = pf[idxmax]
        plot_pf(return_pf)
        param_dict = dict(zip(['window', 'lower', 'upper'], [int(idxmax[0]), round(idxmax[1], 2), round(idxmax[2], 2)]))
        return param_dict, return_pf

def update(price, strategy_param:dict):
    """
        update the strategy with the param dictiorary saved in portfolio
    """
    mom_indicator = get_MomInd().run(price, 
                        window=strategy_param['window'],
                        lower=strategy_param['lower'],
                        upper=strategy_param['upper'],
                        param_product=True)
    entries = mom_indicator.entry_signal
    exits = mom_indicator.exit_signal
    pf = vbt.Portfolio.from_signals(price, entries, exits, fees=0.002, freq='1D')
    return pf