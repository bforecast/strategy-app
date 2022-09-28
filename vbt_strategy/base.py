import pandas as pd

import streamlit as st

from utils.processing import AKData
from utils.vbt_nb import plot_pf



class BaseStrategy(object):
    '''base strategy'''
    _name = "base"
    param_dict ={}
    param_def ={}

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

    def maxSR(self, param, output_bool=False):
        self.param_dict = param
        self.run(output_bool)
        if output_bool:
            plot_pf(self.pf)
       
        return True

    def update(self, param_dict:dict):
        """
            update the strategy with the param dictiorary saved in portfolio
        """
        self.param_dict = {}
        for k, v in param_dict.items():
            self.param_dict[k] = [v]

        self.run()
        return self.pf    

