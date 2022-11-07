import pandas as pd
import numpy as np

import streamlit as st
import vectorbt as vbt

import plotly.express as px
import humanize

from utils.component import  check_password, input_dates
from utils.dataroma import *

from utils.riskfolio import show_OpMS, show_OpMSC
from utils.portfolio import Portfolio
from utils.vbt import get_pfByWeight, get_pfByMaxReturn, plot_pf

@st.cache
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
        "strategy_param": {},
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
                    max_dict['param_dict'] = strategy.param_dict
    # bobs.append(max_dict)
    if fund_desc != "":
        portfolio = Portfolio()
        if portfolio.add(symbolsDate_dict, max_dict['strategy name'], max_dict['param_dict'], max_dict['pf'], fund_desc):
            max_dict['savetodb'] = True
    # limited by the multiprocessing's return error
    if max_dict['pf'] is not None:
        max_dict['pf'] = max_dict['pf'].value()

    return max_dict

def cal_beststrategy(market, symbols, start_date, end_date, fund_desc):
    symbolsDate_dict['market'] = market
    symbolsDate_dict['symbols'] = symbols
    symbolsDate_dict['start_date'] = start_date
    symbolsDate_dict['end_date'] = end_date

    bobs = []
    # update_bar = st.progress(0)
    info_holder = st.empty()
    progress_holder = st.empty()
    i = 0
    for symbol in symbols:
        info_holder.write(f"Calculate symbol('{symbol}')")
        progress_holder.progress(i / (len(symbols)-1))
        symbolsDate_dict['symbols'] = [symbol]
        bobs.append(get_bobmaxsr(symbolsDate_dict.copy(), fund_desc))
        i+=1

    info_holder.empty()
    progress_holder.empty()
    return bobs

if check_password():
    # List of funds to analyze
    funds_tickers = ["BRK", "MKL", "GFT", "psc", "LMM", "oaklx", "ic", "DJCO", "TGM",
                    "AM", "aq", "oc", "HC", "SAM", "PI", "DA", "BAUPOST", "FS", "GR"]
    
    fund_ticker = st.sidebar.selectbox("Select US Funds", funds_tickers, help="data from dataroma.com")
    # si_df = getSuperInvestors()
    # st.write(si_df)
    # selected_df = show_siTable(si_df)
    # if show_siTable(si_df):
    #     fund_ticker = selected_df.loc[0, 'ticker']
    fund_data = getData(fund_ticker)
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

    st.dataframe(df[['Stock', 'Portfolio (%)', 'Recent Activity', 'Reported Price', 'CurrentPrice', 'Reported Price Change (%)']]
                    .style.format({'Portfolio (%)':'{0:,.2f}', 'Reported Price Change (%)':'{0:,.3f}'}), 
                    )
    fig = px.pie(df.iloc[0:10], values='Portfolio (%)', names='Ticker', title='Top 10 holdings')
    st.plotly_chart(fig)
        
    start_date, end_date = input_dates()
    df['Ticker'] = df['Ticker'].apply(lambda x: x.replace('.', '_'))
    symbolsDate_dict={
                    'market':       'US',
                    'symbols':      df.iloc[0:10]['Ticker'].values,
                    'weights':      df.iloc[0:10]['Portfolio (%)'].values,
                    'start_date':   start_date,
                    'end_date':     end_date,
                    }
    subpage = st.sidebar.radio("Option", ('Max Sharpe Portfolio Weights', 'Max Sharpe Stocks First'))
    st.subheader(subpage)
    if subpage == 'Max Sharpe Portfolio Weights':
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
        rm = st.selectbox('Select Risk Measures', rms_dict.keys(), format_func=lambda x: x+' ('+ rms_dict[x]+ ')')
        show_OpMSC(symbolsDate_dict, rm)
    else:
        savetodb = st.checkbox("Save to db")
        desc_str = f"Owned by {fund_ticker} in {fund_data[1]}" if savetodb else ""
        bobs = cal_beststrategy(symbolsDate_dict['market'],
                            symbolsDate_dict['symbols'],
                            symbolsDate_dict['start_date'],
                            symbolsDate_dict['end_date'],
                            desc_str,
                            )
        returns_df = pd.DataFrame()
        for bob_dict in bobs:
            returns_df[bob_dict['symbol']] = bob_dict['pf']
        st.line_chart(returns_df)
        # show_OpMS(returns_df, rm="MV")
        st.write("**Original Fund weights' Portfolio of BOBs**")
        weights = []
        for symbol in returns_df.columns:
            weights.append(df.loc[symbol,'Portfolio (%)'])
        weights = weights / sum(weights)
        pf = get_pfByWeight(returns_df, weights)
        plot_pf(pf, select=False)
        st.write("**Max Return Portfolio of BOBs**")
        pf = get_pfByMaxReturn(returns_df)
        plot_pf(pf, select=False)


