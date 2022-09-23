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

    def __init__(self, symbolsDate_dict:dict):
        market = symbolsDate_dict['market']
        symbols = symbolsDate_dict['symbols']
        start_date = symbolsDate_dict['start_date']
        end_date = symbolsDate_dict['end_date']
        Data = AKData(market)
        self.price_df = pd.DataFrame()
        for symbol in symbols:
            if symbol!='':
                stock_df = Data.download(symbol, start_date, end_date)
                if stock_df.empty:
                    st.warning(f"Warning: stock '{symbol}' is invalid or missing. Ignore it", icon= "⚠️")
                else:
                    self.price_df[symbol] = stock_df['close']
        self.symbols = self.price_df.columns()
        print(self.symbols)

    def log(self, txt, dt=None, doprint=False):
        pass

    def maxSR(self):
        pass

    def update(self, strategy_param:dict):
        pass

