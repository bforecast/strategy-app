import streamlit as st

from utils.component import input_SymbolsDate, button_SavePortfolio, check_password, params_selector
from utils.plot import plot_pf
from vbt_strategy.MA import MAStrategy

def show_PortforlioDetail(strategy):
    pf = strategy.pf
    param_dict = strategy.param_dict
    total_return = round(pf.stats('total_return')[0]/100.0, 2)
    lastday_return = round(pf.returns()[-1], 2)

    sharpe_ratio = round(pf.stats('sharpe_ratio')[0], 2)
    maxdrawdown = round(pf.stats('max_dd')[0]/100.0, 2)
    annual_return = pf.annualized_return()

    cols = st.columns(4 + len(param_dict))
    with cols[0]:
        st.metric('Annualized', "{0:.0%}".format(annual_return))
    with cols[1]:
        st.metric('Lastday Return', "{0:.1%}".format(lastday_return))
    with cols[2]:
        st.metric('Sharpe Ratio', '%.2f'% sharpe_ratio)
    with cols[3]:
        st.metric('Max DD', '{0:.0%}'.format(maxdrawdown))
    i = 4
    for k, v in param_dict.items():
        with cols[i]:
            st.metric(k, v)
        i = i + 1


if check_password():
    symbolsDate_dict = input_SymbolsDate()
    if len(symbolsDate_dict['symbols']) > 0:
        st.header(f"{symbolsDate_dict['symbols']} Strategies' comparision board")
        strategy_list = getattr(__import__(f"vbt_strategy"), 'strategy_list')
        for strategyname in strategy_list:
            strategy_cls = getattr(__import__(f"vbt_strategy"), strategyname + 'Strategy')
            strategy = strategy_cls(symbolsDate_dict)
            if len(strategy.ohlcv_list) > 0:
                if strategy.maxSR(strategy.param_dict, output_bool=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"Strategy '{strategyname}' Max Sharpe_Ratio Result")
                    with col2:
                        showpf_bool = st.checkbox("Show the Portfolio", key='checkbox_'+strategyname)
                    
                    show_PortforlioDetail(strategy)
                    if showpf_bool:
                        plot_pf(strategy.pf)
                    savepf_bool = button_SavePortfolio(symbolsDate_dict, strategyname, strategy.param_dict, strategy.pf)
            else:
                st.error("None of stocks is valid.")

