import streamlit as st

from utils.component import input_SymbolsDate, button_SavePortfolio, check_password
from vbt_strategy.MOM import MOMStrategy

if check_password():
    symbolsDate_dict = input_SymbolsDate()
    
    if len(symbolsDate_dict['symbols']) > 0:
        st.header("Strategy MOM") 
        strategy = MOMStrategy(symbolsDate_dict)
        if len(strategy.ohlcv_list) > 0:
            param_dict,pf = strategy.maxSR(output_bool=True)
            button_SavePortfolio(symbolsDate_dict, 'MOM', param_dict, pf)
        else:
            st.error("None of stocks is valid.")
