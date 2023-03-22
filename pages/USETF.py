import pandas as pd
import numpy as np
import json

import streamlit as st
import vectorbt as vbt

import plotly.express as px
import humanize

from utils.component import  check_password, input_dates
from utils.dataroma import *

from utils.riskfolio import get_pfOpMS, FactorExposure, plot_AssetsClusters
from utils.portfolio import Portfolio
from utils.vbt import get_pfByWeight, get_pfByMaxReturn, plot_pf
from utils.processing import get_stocks
from utils.rrg import plot_RRG
from pages.SuperInvestors import *

@st.cache_data(ttl = 86400)
def getETFData(holding_ticker):
    # Data Extraction
    # We obtain the HTML from the corresponding fund in Dataroma.

    html = requests.get(
        "https://etfdb.com/etf/" + holding_ticker + "/#holdings", headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:94.0) Gecko/20100101 Firefox/94.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "es-ES,es;q=0.8,en-US;q=0.5,en;q=0.3",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0"
        }).content

    # Non-table Data Parsing
    soup = BeautifulSoup(html, "html.parser")
    name = soup.find('title').text
    portfolio_date = soup.find('time',class_="date-modified").text

    # Table Data Parsing
    df_list = pd.read_html(html)
    for df in df_list:
        if('Symbol Symbol' in df.columns):
            df = df.head(15)
            break

    # Column name corrections.
    df = df.rename(columns={"Symbol Symbol": "Ticker"})
    df = df.rename(columns={"Holding Holding": "Stock"})
    df = df.rename(columns={"% Assets % Assets": "Portfolio (%)"})

    df['Ticker'] = df['Ticker'].apply(lambda x: x.replace('.', '_'))
    df["Portfolio (%)"] = df["Portfolio (%)"].apply(lambda x: pd.to_numeric(x.split("%")[0]))
    df.index = df['Ticker']

    return [name, None, portfolio_date, df]

def main1():
    # 1. display selected fund's information
    # List of funds to analyze
    funds_tickers = ["QQQ", "SPY", "XLB", "XLC", 
                     "XLE", "XLF", "XLI", "XLK", 
                     "XLP", "XLRE", "XLU", "XLV", 
                     "XLY", "XHB", "SDY", "SMH", 
                     "ARKK", "ARKW", "ARKQ", 
                     "ARKF", "ARKG", "ARKX", 
                     "XME", "XPH", "KBE"]

    fund_ticker = st.sidebar.selectbox("Select US ETF Funds", funds_tickers, help="data from ETF provider", key='USETF')
    try:
        fund_data = getETFData(fund_ticker)
    except ValueError as ve:
        st.write(f"etfdata-getETFData error: {ve}")
        return
        
    # Fund positions
    df = fund_data[-1]
    st.subheader(fund_data[0])
    col1, col2, col3, col4 = st.columns(4)
    with col1:
            st.write('**Portfolio_date:**')
    with col2:
            st.write(fund_data[2])
    with col3:
            st.write('**Top 10 holdings(%):**')
    with col4:
            st.write(round(df["Portfolio (%)"].iloc[0:10].sum(),2))

    with st.expander("Portfolio Table (Top 15)"):
        st.dataframe(df[['Stock', 'Portfolio (%)']]
                        .style.format({'Portfolio (%)':'{0:,.2f}'})
                        .background_gradient(cmap='YlGn'), 
                    )

    # 2.select optimized portfolio strategies.
    start_date, end_date = input_dates(by='USETF')
    symbolsDate_dict={
                    'market':       'US',
                    'symbols':      df.iloc[0:10]['Ticker'].tolist(),
                    'weights':      df.iloc[0:10]['Portfolio (%)'].tolist(),
                    'start_date':   start_date,
                    'end_date':     end_date,
                    }
    benchmark_dict = {
                   'market':       'US',
                   'symbols':      ['SPY'],
                   'weights':      [ 1.0 ],
                   'start_date':   start_date,
                   'end_date':     end_date,
                   }

    subpage = st.sidebar.radio("Select Optimized Methods:", ('Original Weights', 'Max Sharpe Weights'), #'Optimize stocks first, then Maximize total return'
                                horizontal=True)
    if subpage == 'Original Weights':
        # 2.1.1 plot Pie chart of Orginial fund porforlio.
        st.subheader(subpage)
        fig = px.pie(df.iloc[0:10], values='Portfolio (%)', names='Ticker', title='Top 10 holdings')
        st.plotly_chart(fig)

        # 2.1.2 plot pf chart of Orginial fund porforlio.
        stocks_df = get_stocks(symbolsDate_dict,'close')
        benchmark_df = get_stocks(benchmark_dict,'close')
        # vbt.settings['portfolio']['stats']['settings']['benchmark'] = 'SPY'
        # vbt.settings['portfolio']['stats']['settings']['benchmark_rets'] = benchmark_df['SPY'].vbt.to_returns()

        weights = []
        for symbol in stocks_df.columns:
            weights.append(df.loc[symbol,'Portfolio (%)'])
        weights = weights / sum(weights)
        pf = get_pfByWeight(stocks_df, weights)

        plot_pf(pf, select=False, name=f"{fund_ticker}-Original Weights", bm_symbol = benchmark_dict['symbols'][0], bm_price = benchmark_df[benchmark_dict['symbols'][0]])

        # 2.1.3 calculate the factors effect of Original fund portfolio.
        show_FactorExposure(symbolsDate_dict, pf, stocks_df)
        # 2.1.4 Assets Clusters of Original fund portfolio.
        st.write("**资产层次聚类(Assets Clusters)：**")
        with st.expander("The codependence or similarity matrix: pearson; Linkage method of hierarchical clustering: ward"):
            plot_AssetsClusters(stocks_df)
        # 2.1.5 plot RRG
        st.write("**相对轮动(RRG):**")
        symbol_benchmark = 'SPY'
        symbolsDate_dict['symbols'] += [symbol_benchmark]
        stocks_df = get_stocks(symbolsDate_dict,'close')
        plot_RRG(symbol_benchmark, stocks_df)
            
    elif subpage == 'Max Sharpe Weights':
        # 2.2.1 calculate the optimized max sharpe ratio's portfolio.
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
        stocks_df = get_stocks(symbolsDate_dict, 'close')
        pf = get_pfOpMS(stocks_df, rm)
        plot_pf(pf, select=False, name=f"{fund_ticker}-Max Sharpe Weights")
        # 2.2.2 calculate the factors effect of optimized max sharpe ratio's portfolio.
        show_FactorExposure(symbolsDate_dict, pf, stocks_df)

    else:
        st.warning("Not available.") 

if check_password():
    main1()