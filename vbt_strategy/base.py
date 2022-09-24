from asyncio.proactor_events import _ProactorBasePipeTransport
import numpy as np
import pandas as pd
from datetime import datetime

import streamlit as st
import vectorbt as vbt

from utils.vbt_nb import plot_pf
from utils.processing import AKData


class BaseStrategy(object):
    '''base strategy'''
    _name = "base"
    param_dict ={}

    def __init__(self, symbolsDate_dict:dict):
        market = symbolsDate_dict['market']
        symbols = symbolsDate_dict['symbols']
        start_date = symbolsDate_dict['start_date']
        end_date = symbolsDate_dict['end_date']
        Data = AKData(market)
        self.price_df = pd.DataFrame()
        self.ohlcv_list = []
        for symbol in symbols:
            if symbol!='':
                stock_df = Data.download(symbol, start_date, end_date)
                if stock_df.empty:
                    st.warning(f"Warning: stock '{symbol}' is invalid or missing. Ignore it", icon= "⚠️")
                else:
                    self.ohlcv_list.append((symbol, stock_df))
        
    def log(self, txt, dt=None, doprint=False):
        pass

    def maxSR(self):
        pass

    def update(self, param_dict:dict):
        """
            update the strategy with the param dictiorary saved in portfolio
        """
        self.param_dict['window'] = [param_dict['window']]
        self.param_dict['upper'] = [param_dict['upper']]
        self.param_dict['lower'] = [param_dict['lower']]

        pf = self.run()
        return pf    

