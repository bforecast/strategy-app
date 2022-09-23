import pandas as pd
import numpy as np
import streamlit as st

import vectorbt as vbt

from utils.processing import AKData
# from utils.db import save_strategy
from utils.component import input_symbols, input_dates, button_SavePortfolio, check_password

from vbt_strategy.PairTrading import PairTrade

if check_password():
    market, symbols = input_symbols()
    if len(symbols) >= 2:
        start_date, end_date = input_dates()
        Data = AKData(market)
        price = pd.DataFrame()
        symbol_list = symbols.copy()
        ohlcv_dict = {}
        for symbol in symbol_list:
            stock_df = Data.download(symbol, start_date, end_date)
            if stock_df.empty:
                st.warning(f"Warning: stock '{symbol}' is invalid or missing. Ignore it", icon= "⚠️")
                symbols.remove(symbol)
            else:
                ohlcv_dict[symbol] = stock_df
                price[symbol] = stock_df['close']
        if len(ohlcv_dict) == 0:
            st.error('None stock left',  icon="🚨")
        else:
            returns = price.pct_change()
            st.line_chart(returns.cumsum())
            param_dict,pf = PairTrade(ohlcv_dict[symbols[0]], ohlcv_dict[symbols[1]],symbols[0], symbols[1])
            button_SavePortfolio(market, symbols[0:2], 'PairTrade', param_dict, pf, start_date, end_date)

    else:
        st.warning('Needs a Pair of Stocks', icon= "ℹ️")

