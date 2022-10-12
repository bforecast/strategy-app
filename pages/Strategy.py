import streamlit as st

from utils.component import input_SymbolsDate, button_SavePortfolio, check_password, params_selector
from vbt_strategy.MA import MAStrategy

def check_params(params):
    # for key, value in params.items():
    #     if len(params[key]) < 2:
    #         st.error(f"{key} 's numbers are not enough. ")
    #         return False
    return True
    
if check_password():
    strategy_list = getattr(__import__(f"vbt_strategy"), 'strategy_list')
    strategyname = st.sidebar.selectbox("Please select strategy", strategy_list)
    if strategyname:
        symbolsDate_dict = input_SymbolsDate()
        if len(symbolsDate_dict['symbols']) > 0:
            st.header(strategyname)
            st.subheader("Stocks:    " + ' , '.join(symbolsDate_dict['symbols']))
            strategy_cls = getattr(__import__(f"vbt_strategy"), strategyname + 'Strategy')
            strategy = strategy_cls(symbolsDate_dict)
            if len(strategy.stock_df) > 0:
                params = params_selector(strategy.param_def)
                if check_params(params):
                    if strategy.maxSR(params, output_bool=True):
                        st.text("Max Sharpe_Ratio's parameters:    " + str(strategy.param_dict))
                        button_SavePortfolio(symbolsDate_dict, strategyname, strategy.param_dict, strategy.pf)
                    else:
                        st.error("Stocks don't match the Strategy")
            else:
                st.error("None of stocks is valid.")

