import streamlit as st

from utils.component import input_SymbolsDate, button_SavePortfolio, check_password
from vbt_strategy.PairTrade import PairTradeStrategy

if check_password():
    symbolsDate_dict = input_SymbolsDate()
    
    if len(symbolsDate_dict['symbols']) > 1:
        st.header("Strategy PairTrading") 
        strategy = PairTradeStrategy(symbolsDate_dict)
        if len(strategy.ohlcv_list) > 1:
            param_dict,pf = strategy.maxSR(output_bool=True)
            button_SavePortfolio(symbolsDate_dict, 'PairTrade', param_dict, pf)
        else:
            st.error("Need 2 stocks for PairTrading")

