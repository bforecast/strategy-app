import streamlit as st

from utils.component import input_SymbolsDate, check_password, form_SavePortfolio
from utils.db import get_symbolname
from utils.vbt import display_pfbrief

if check_password():
    symbolsDate_dict = input_SymbolsDate()
    if len(symbolsDate_dict['symbols']) > 0:
        st.header(f"{get_symbolname(symbolsDate_dict['symbols'][0])} Strategies' comparision board")
        strategy_list = getattr(__import__(f"vbt_strategy"), 'strategy_list')
        for strategyname in strategy_list:
            strategy_cls = getattr(__import__(f"vbt_strategy"), strategyname + 'Strategy')
            strategy = strategy_cls(symbolsDate_dict)
            if len(strategy.stock_dfs) > 0:
                if strategy.maxSR(strategy.param_dict, output_bool=False):
                    # col1, col2 = st.columns(2)
                    # with col1:
                    st.info(f"Strategy '{strategyname}' Max Sharpe_Ratio Result")
                    # with col2:
                    #     showpf_bool = st.checkbox("Show the Portfolio", key='checkbox_'+strategyname)
                    display_pfbrief(strategy)
                    # if showpf_bool:
                    #     plot_pf(strategy.pf)
                    form_SavePortfolio(symbolsDate_dict, strategyname, strategy.param_dict, strategy.pf)
            else:
                st.error("None of stocks is valid.")

