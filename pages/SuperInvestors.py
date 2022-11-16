import pandas as pd
import numpy as np
import json

import streamlit as st
import vectorbt as vbt

import plotly.express as px
import humanize

from utils.component import  check_password, input_dates
from utils.dataroma import *

from utils.riskfolio import show_OpMS
from utils.portfolio import Portfolio
from utils.vbt import get_pfByWeight, get_pfByMaxReturn, plot_pf
from utils.processing import get_stocks

@st.cache(allow_output_mutation=True)
def get_bobmaxsr(symbolsDate_dict:dict, fund_desc:str = ""):
    '''
    get the best of best max sharpe ratio solution
    '''
    strategy_list = getattr(__import__(f"vbt_strategy"), 'strategy_list')
    max_dict = {
        "symbol": symbolsDate_dict['symbols'][0],
        "sharpe ratio": 0,
        "pf":   None,
        "strategy name": '',
        "param_dict": {},
        "savetodb": False
        }
    for strategyname in strategy_list:
        strategy_cls = getattr(__import__(f"vbt_strategy"), strategyname + 'Strategy')
        strategy = strategy_cls(symbolsDate_dict)
        if len(strategy.stock_dfs) > 0:
            if strategy.maxSR(strategy.param_dict, output_bool=False):
                sharpe_ratio = round(strategy.pf.stats('sharpe_ratio')[0], 2)
                if sharpe_ratio > max_dict['sharpe ratio']:
                    max_dict['sharpe ratio'] = sharpe_ratio
                    max_dict['pf'] = strategy.pf
                    max_dict['strategy name'] = strategyname
                    max_dict['param_dict'] = strategy.param_dict.copy()
    return max_dict

def cal_beststrategy(symbolsDate_dict, fund_desc):
    bobs = []
    info_holder = st.empty()
    expander_holder = st.expander("Best Strategy of Top 10 Stocks", expanded=True)
    sd_dict = symbolsDate_dict.copy()
    portfolio = Portfolio()
    with expander_holder:
        col1, col2, col3, col4, col5 = st.columns((1, 1, 1, 6, 1))
        col1.text('Symbol')  
        col2.text('Sharpe Ratio')
        col3.text('Strategy')
        col4.text('Parameters')
    for symbol in symbolsDate_dict['symbols']:
        info_holder.write(f"Calculate symbol('{symbol}')")
        sd_dict['symbols'] = [symbol]
        bob_dict = get_bobmaxsr(sd_dict, fund_desc)
        bobs.append(bob_dict)
        with expander_holder:
            col1, col2, col3, col4, col5 = st.columns((1, 1, 1, 6, 1))
            col1.text(bob_dict['symbol'])  
            col2.text(bob_dict['sharpe ratio'])
            col3.text(bob_dict['strategy name'])
            col4.text(json.dumps(bob_dict['param_dict']),)
            button_type = "Save"
            button_phold = col5.empty()  # create a placeholder
            do_action = button_phold.button(button_type, key='btn_save_'+symbol)
            if do_action:
                if portfolio.add(sd_dict, bob_dict['strategy name'], bob_dict['param_dict'], bob_dict['pf'], fund_desc):
                    button_phold.write("Saved")
                else:
                    button_phold.write('Fail')

    info_holder.empty()
    # expander_holder.expander = False
    return bobs

def main():
    # 1. display selected fund's information
    # List of funds to analyze
    funds_tickers = ["BRK", "MKL", "GFT", "psc", "LMM", "oaklx", "ic", "DJCO", "TGM",
                    "AM", "aq", "oc", "HC", "SAM", "PI", "DA", "BAUPOST", "FS", "GR"]
    
    fund_ticker = st.sidebar.selectbox("Select US Funds", funds_tickers, help="data from dataroma.com")
    # si_df = getSuperInvestors()
    # st.write(si_df)
    # selected_df = show_siTable(si_df)
    # if show_siTable(si_df):
    #     fund_ticker = selected_df.loc[0, 'ticker']
    try:
        fund_data = getData(fund_ticker)
    except ValueError as ve:
        st.write(f"dataroma-getData error: {ve}")
        return
        
    # Fund positions
    df = fund_data[-1]
    st.subheader(fund_data[0])
    col1, col2, col3, col4 = st.columns(4)
    with col1:
            st.write('**Period:**')
            st.write('**Portfolio_date:**')
            st.write('**Total value:**')
    with col2:
            st.write(fund_data[1])
            st.write(fund_data[2])
            st.write(humanize.intword(df["Value"].sum()))
    with col3:
            st.write('**Num_of_positions:**')
            st.write('**Top 10 holdings(%):**')
            st.write('**Price Change (%)**')
    with col4:
            st.write(df["Stock"].count())
            st.write(round(df["Portfolio (%)"].iloc[0:10].sum(),2))
            st.write(round(df["Reported Price Change (%)"].mean(), 2))

    with st.expander("Portfolio Table"):
        st.dataframe(df[['Stock', 'Portfolio (%)', 'Recent Activity', 'Reported Price', 'CurrentPrice', 'Reported Price Change (%)']]
                        .style.format({'Portfolio (%)':'{0:,.2f}', 'Reported Price Change (%)':'{0:,.3f}'})
                        .background_gradient(cmap='YlGn'), 
                    )


    # 2.select optimized portfolio strategies.
    start_date, end_date = input_dates()
    symbolsDate_dict={
                    'market':       'US',
                    'symbols':      df.iloc[0:10]['Ticker'].values,
                    'weights':      df.iloc[0:10]['Portfolio (%)'].values,
                    'start_date':   start_date,
                    'end_date':     end_date,
                    }
    subpage = st.radio("Performance", ('Original Weights', 'Max Sharpe Weights', 'Optimize stocks first, then Maximize total return'), 
                        label_visibility='hidden', horizontal=True)
    if subpage == 'Original Weights':
        st.subheader(subpage)
        fig = px.pie(df.iloc[0:10], values='Portfolio (%)', names='Ticker', title='Top 10 holdings')
        st.plotly_chart(fig)

        stock_dfs = get_stocks(symbolsDate_dict)
        stocks_df = pd.DataFrame()
        for stock_tuple in stock_dfs:
            stocks_df[stock_tuple[0]] = stock_tuple[1].close
        weights = []
        for symbol in stocks_df.columns:
            weights.append(df.loc[symbol,'Portfolio (%)'])
        weights = weights / sum(weights)

        pf = get_pfByWeight(stocks_df, weights)
        plot_pf(pf, select=False, name=f"{fund_ticker}-Original Weights")

    elif subpage == 'Max Sharpe Weights':
        rms_dict = {
                'MV': "Standard Deviation",
                'MAD': "Mean Absolute Deviation",
                'MSV': "Semi Standard Deviation",
                'FLPM': "First Lower Partial Moment (Omega Ratio)",
                'SLPM': "Second Lower Partial Moment (Sortino Ratio)",
                'CVaR': "Conditional Value at Risk",
                'EVaR': "Entropic Value at Risk",
                'WR':   "Worst Realization (Minimax)",
                'MDD': "Maximum Drawdown of uncompounded cumulative returns (Calmar Ratio)",
                'ADD': "Average Drawdown of uncompounded cumulative returns",
                'CDaR': "Conditional Drawdown at Risk of uncompounded cumulative returns",
                'EDaR': "Entropic Drawdown at Risk of uncompounded cumulative returns",
                'UCI': "Ulcer Index of uncompounded cumulative returns",
                }
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(subpage)
        with col2:
            rm = st.selectbox('Select Risk Measures', rms_dict.keys(), 
                            format_func=lambda x: x+' ('+ rms_dict[x]+ ')')
        stock_dfs = get_stocks(symbolsDate_dict)
        stocks_df = pd.DataFrame()
        for stock_tuple in stock_dfs:
            stocks_df[stock_tuple[0]] = stock_tuple[1].close

        show_OpMS(stocks_df, rm, plotname=f"{fund_ticker}-Max Sharpe Weights")
    else:
        st.subheader(subpage)
        bobs = cal_beststrategy(symbolsDate_dict, f"Owned by {fund_ticker} in {fund_data[1]}")
        returns_df = pd.DataFrame()
        bobs_df = pd.DataFrame()
        for bob_dict in bobs:
            returns_df[bob_dict['symbol']] = bob_dict['pf'].value()
            bobs_df = bobs_df.append({'symbol': bob_dict['symbol'],
                                    'strategy name': bob_dict['strategy name'],
                                    'param_dict': json.dumps(bob_dict['param_dict']),
                                    'sharpe ratio': bob_dict['sharpe ratio']
                                    }, ignore_index=True)
        st.write("**Step 1: Original Fund weights' Portfolio of BOBs**")
        weights = []
        for symbol in returns_df.columns:
            weights.append(df.loc[symbol,'Portfolio (%)'])
        weights = weights / sum(weights)
        pf = get_pfByWeight(returns_df, weights)

        pie_df = pd.DataFrame({'Ticker':returns_df.columns, 'Weights': weights})
        st.plotly_chart(
            px.pie(pie_df, values='Weights', names='Ticker', title="Max Return's Allocation")
            )
        plot_pf(pf, select=False, name=f"{fund_ticker}-Top10 Stocks max sharpe -Original Weights")

        st.write("**Step 2: Max Return Portfolio of BOBs**")
        pf, newWeights = get_pfByMaxReturn(returns_df)
        pie_df = pd.DataFrame({'Ticker':returns_df.columns, 'Weights': newWeights})
        st.plotly_chart(
            px.pie(pie_df, values='Weights', names='Ticker', title="Max Return's Allocation")
            )
        plot_pf(pf, select=False, name=f"{fund_ticker}-Top10 Stocks max sharpe -Max return Weights")

if check_password():
    main()
