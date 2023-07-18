import pandas as pd
import numpy as np
import pytz
from datetime import datetime, date
import json

import streamlit as st

st.set_page_config(page_title="BForecast Strategy App")

import vectorbt as vbt

from utils.vbt import plot_pf
from utils.component import check_password, params_selector
from utils.portfolio import Portfolio, selectpf_bySymbols
from vbt_strategy.PairTrade import pairtrade_pfs
from pages.Strategy import check_params
from utils.db import get_SymbolName
from utils.vbt import display_pfbrief

import config

def select_portfolios(portfolio_df, update_bokeh=True):
        df_with_selections = portfolio_df.copy()
        df_with_selections.set_index('id', inplace=True)
        df_with_selections.insert(0, "Select", False)
        # display in 100% percentage format
        df_with_selections['annual_return'] *= 100
        df_with_selections['lastday_return'] *= 100
        df_with_selections['total_return'] *= 100
        df_with_selections['maxdrawdown'] *= 100

        edited_df = st.data_editor(
                        df_with_selections,
                        hide_index=True,
                        use_container_width=True,
                        column_order=['Select','name', 'annual_return','lastday_return', 'sharpe_ratio', 'total_return', 'maxdrawdown', 'symbols', 'end_date'],
                        column_config={
                                "Select":           st.column_config.CheckboxColumn(required=True, width='small'),
                                "sharpe_ratio":     st.column_config.Column(width='small'),
                                "annual_return":    st.column_config.NumberColumn(required=True, format='%i%%', width='small'),
                                "lastday_return":    st.column_config.NumberColumn(required=True, format='%.1f%%', width='small'),    
                                "total_return":    st.column_config.NumberColumn(required=True, format='%i%%', width='small'),    
                                "maxdrawdown":    st.column_config.NumberColumn(required=True, format='%i%%', width='small'),        
                            },
                        disabled=['name', 'annual_return','lastday_return', 'sharpe_ratio', 'total_return', 'maxdrawdown', 'symbols', 'end_date'],
                    )
        selected_ids = list(edited_df[edited_df.Select].index)
        return selected_ids

def show_PortfolioTable(portfolio_df):
    ## using new st.data_editor
    def stringlist_to_set(strlist: list):
        slist = []
        for sstr in strlist:
            for s in sstr.split(','):
                slist.append(s)
        slist = list(dict.fromkeys(slist))
        slist.sort()
        return(slist)
    
    symbols = stringlist_to_set(portfolio_df['symbols'].values)
    if 'symbolsSel' not in st.session_state:
        st.session_state['symbolsSel'] = symbols

    sSel = st.multiselect("Please select symbols:", symbols, 
                                format_func=lambda x: get_SymbolName(x)+x,
                                help='empty means all')
    if st.session_state['symbolsSel'] == sSel:
        update_bokeh = False
    else:
        if len(sSel) > 0:
            if st.session_state['symbolsSel'] == sSel:
                update_bokeh = False
            else:
                st.session_state['symbolsSel'] = sSel
                update_bokeh = True
        else:
            if st.session_state['symbolsSel'] == symbols:
                update_bokeh = False
            else:
                st.session_state['symbolsSel'] = symbols
                update_bokeh = True

    update_bokeh = (update_bokeh or st.session_state['update_bokeh'])
    df = selectpf_bySymbols(portfolio_df, st.session_state['symbolsSel'])
    selectpf = select_portfolios(df, update_bokeh)
    st.session_state['update_bokeh'] = False
    return(selectpf)

def show_PortforlioDetail(portfolio_df, index):
    if index > -1 and (index in portfolio_df.index):
        st.info('Selected portfolio:    ' + portfolio_df.at[index, 'name'])
        param_dict = json.loads(portfolio_df.at[index, 'param_dict'])
        display_pfbrief(pf=vbt.Portfolio.loads(portfolio_df.at[index, 'vbtpf']), param_dict=param_dict)
        st.markdown("**Description**")
        st.markdown(portfolio_df.at[index, 'description'], unsafe_allow_html=True)
        return True
    else:
        return False

def show_PortforlioYearly(pf_row):
    end_date = date.today()
    end_date = datetime(year=end_date.year, month=end_date.month, day=end_date.day, tzinfo=pytz.utc)
    symbolsDate_dict = {
            "market":   pf_row['market'],
            "symbols":  [],
            "start_date": end_date,
            "end_date": end_date,
        }
    
    strategyname = pf_row['strategy']
    
    # get the strategy class according to strategy name
    strategy_cli = getattr(__import__(f"vbt_strategy"), f"{strategyname}Strategy")
    strategy = strategy_cli(symbolsDate_dict)
    params = params_selector(strategy.param_def)

    pfYearly_df = pd.DataFrame()                                 
    for y in range(1, 10, 2):
        start_date = datetime(year=end_date.year-y, month=end_date.month, day=end_date.day, tzinfo=pytz.utc)
        st.write(start_date)
        symbolsDate_dict = {
            "market":   pf_row['market'],
            "symbols":  [pf_row['symbols']],
            "start_date": start_date,
            "end_date": end_date,
        }
        strategy = strategy_cli(symbolsDate_dict)

        if check_params(params):
            if strategy.maxRARM(params, output_bool=False):
                st.text("Max Sharpe_Ratio's parameters:    " + str(strategy.param_dict))
                pfYearly_df = pfYearly_df.append({
                            "year": y,
                            'total_return': round(strategy.pf.stats('total_return')[0]/100.0, 2),
                            'lastday_return': round(strategy.pf.returns()[-1], 2),
                            'sharpe_ratio':  round(strategy.pf.stats('sharpe_ratio')[0], 2),
                            'maxdrawdown':   round(strategy.pf.stats('max_dd')[0]/100.0, 2),
                            'annual_return': round(strategy.pf.annualized_return(), 2)
                        }, ignore_index=True, )
        
    st.table(pfYearly_df)    


def main():
    st.header("Portfolio Board")
    selected_pfs = []
    portfolio = Portfolio()
    if 'update_bokeh' not in st.session_state:
        st.session_state['update_bokeh'] = True # update and refresh bokeh table
    selected_pfs = show_PortfolioTable(portfolio.df)
    if len(selected_pfs) > 1:
        ##多portfolio比较
        value_df = pd.DataFrame()
        position_df = pd.DataFrame()
        for index in selected_pfs:
            pf = vbt.Portfolio.loads(portfolio.df.loc[index, 'vbtpf'])
            value_df[portfolio.df.loc[index, 'name']] = pf.value()
            position_df[portfolio.df.loc[index, 'name']] = pf.position_mask()
            show_PortforlioDetail(portfolio.df, index)
        # value_df = value_df.cumsum()
        value_df.fillna(method='ffill', inplace=True)
        # value_df['mean'] = value_df.mean(axis=1)
        st.line_chart(value_df)
        st.plotly_chart(position_df.vbt.plot(), user_container_width = True)

        if len(selected_pfs) == 2:
            if st.button("PairTrade"):
                pf = pairtrade_pfs(value_df.columns[0], value_df.columns[1],
                                value_df.iloc[:,0], value_df.iloc[:,1], True)
                if pf:
                    plot_pf(pf, name="PairTrade--"+value_df.columns[0]+'&'+value_df.columns[1])
                else:
                    st.error(f"Fail to PairTrade '{value_df.columns[0]}' & '{value_df.columns[1]}'.")
            
            if st.button("Master/Backup"):
                    symbol1 = value_df.columns[0]
                    symbol2 = value_df.columns[1]

                    symbol_cols = pd.Index([symbol1, symbol2], name='symbol')
                    # Build percentage order size
                    vbt_order_size = pd.DataFrame(index=value_df.index, columns=symbol_cols)
                    vbt_order_size[symbol1] = 0
                    vbt_order_size[symbol2] = 1
                    vbt_order_size.loc[position_df.iloc[:,0], symbol1] = 1
                    vbt_order_size.loc[position_df.iloc[:,0], symbol2] = 0
                    # Execute at the next bar
                    vbt_order_size = vbt_order_size.vbt.fshift(1)

                    vbt_close_price = pd.concat((value_df.iloc[:,0], value_df.iloc[:,1]), axis=1, keys= symbol_cols)

                    pf_kwargs = dict(fees=0.001, freq='1D')
                    ms_pf = vbt.Portfolio.from_orders(
                                                    vbt_close_price,  # current close as reference price
                                                    size=vbt_order_size,  
                                                    price=vbt_close_price,  # current open as execution price
                                                    size_type='targetpercent', 
                                                    init_cash=100,
                                                    cash_sharing=True,  # share capital between assets in the same group
                                                    group_by=True,  # all columns belong to the same group
                                                    call_seq='auto',  # sell before buying
                                                    **pf_kwargs)
                    plot_pf(ms_pf, name="Master/Backup--"+symbol1 + '&' + symbol2)
                    value_df["Master/Backup"] = ms_pf.value()
                    st.line_chart(value_df)



    elif len(selected_pfs) == 1 :
        ##单portforlio详情
        index = selected_pfs[0]      
        if show_PortforlioDetail(portfolio.df, index):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                showpf_bool = st.checkbox("Show the Portfolio")
            with col2:
                morepf_bool = st.button('More')      
            with col3:
                updatepf_bool = st.button('Update')
            with col4:
                deletepf_bool = st.button('Delete')

            if showpf_bool:
                try:
                    pf = vbt.Portfolio.loads(portfolio.df.loc[index, 'vbtpf'])
                    plot_pf(pf, name=portfolio.df.loc[index, 'name'])
                except ValueError as ve:
                    print(f"show_PortforlioDetail:{portfolio.df.loc[index,'name']} error --{ve}")
                    st.error('Fail to load pf.')
                    
            if morepf_bool:
                show_PortforlioYearly(portfolio.df.loc[index, :])
            if updatepf_bool:
                if portfolio.update(index):
                    st.success('Update portfolio Sucessfully.')
                    st.experimental_rerun()
                else:
                    st.error('Fail to update portfolio.')
                st.session_state['update_bokeh'] = True
                # st.experimental_rerun()

            if deletepf_bool:
                if portfolio.delete(portfolio.df.loc[index, 'id']):
                    st.success('Delete portfolio Sucessfully.')
                else:
                    st.error('Fail to delete portfolio.')
                st.session_state['update_bokeh'] = True
                st.experimental_rerun()
        else:
            selected_pfs = []

    else:
        ##无选择portforlio
        if st.button("Update All"):
            update_bar = st.progress(0)
            num_portfolio = len(portfolio.df)
            info_holder = st.empty()
            for i in range(num_portfolio):
                info_holder.write(f"updating portfolio('{portfolio.df.iloc[i]['name']}')")
                if not portfolio.update(portfolio.df.iloc[i]['id']):
                    st.error(f"Fail to update portfolio('{portfolio.df.iloc[i]['name']}')")
                update_bar.progress(i / (num_portfolio-1))
                info_holder.empty()

            st.experimental_rerun()

if __name__ == "__main__":
    if check_password():
        main()
