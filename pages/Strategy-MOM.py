# import pandas as pd
# import numpy as np
import streamlit as st

# import vectorbt as vbt

from utils.processing import AKData
from utils.component import input_symbols, input_dates, button_SavePortfolio, check_password
from vbt_strategy.MOM import MOM_MaxSR

if check_password():
    market, symbols = input_symbols()
    start_date, end_date = input_dates()
    Data = AKData(market)
    for symbol in symbols:
        if symbol!='':
            stock_df = Data.download(symbol, start_date, end_date)
            if stock_df.empty:
                st.warning(f"Warning: stock '{symbol}' is invalid or missing. Ignore it", icon= "⚠️")
            else:
                param_dict,pf = MOM_MaxSR(symbol, stock_df['close'])
                button_SavePortfolio(market, [symbol], 'MOM', param_dict, pf, start_date, end_date)
