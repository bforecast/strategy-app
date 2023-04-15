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
from utils.rrg import plot_RRG, RRG_Strategy

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
# @st.cache_data()
def get_bobmaxsr(_symbolsDate_dict:dict, fund_desc:str = ""):
    '''
    get the best of best max sharpe ratio solution
    '''
    strategy_list = getattr(__import__(f"vbt_strategy"), 'strategy_list')
    max_dict = {
        "symbol": _symbolsDate_dict['symbols'][0],
        "sharpe ratio": 0,
        "pf":   None,
        "strategy name": '',
        "param_dict": {},
        "savetodb": False
        }
    for strategyname in strategy_list:
        strategy_cls = getattr(__import__(f"vbt_strategy"), strategyname + 'Strategy')
        strategy = strategy_cls(_symbolsDate_dict)
        if len(strategy.stock_dfs) > 0:
            strategy.param_dict['RARM'] = 'sharpe_ratio'
            strategy.param_dict['WFO'] = 'None'
            if strategy.maxRARM(strategy.param_dict, output_bool=False):
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
            print(bob_dict['param_dict'])
            col4.text(json.dumps(bob_dict['param_dict']))
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

def show_FactorExposure(symbolsDate_dict, pf, stocks_df):
    factors_dict = {
        "None" : "Select the Factors Exposure",
        "iShares 5 factors" : {
            'MTUM' : "动量", 
            'QUAL' : "质量", 
            'SIZE' : "规模", 
            'USMV' : "低波动", 
            'VLUE' : "价值"
        },
        "All Sector factors" : {
            "XLB" : "原材料",
            "XLC" : "通讯",
            "XLE" : "能源",
            "XLF" : "金融",
            "XLI" : "工业制造",
            "XLK" : "技术",
            "XLP" : "必需消费",
            "XLRE": "房地产",
            "XLU" : "公用事业",
            "XLV" : "医药",
            "XLY" : "可选消费品"
        }
    }
    factors_sel = st.selectbox("**因子暴露分析(Factor Exposures)**", factors_dict.keys(), label_visibility='collapsed')
    factors_sel = factors_dict[factors_sel]
    if isinstance(factors_sel, dict):
        st.write('、'.join(k+'('+v+')' for v,k in factors_sel.items()))
        sd_dict = symbolsDate_dict.copy()
        sd_dict['symbols'] = factors_sel.keys()
        factors_df = get_stocks(sd_dict, 'close')
        portfolio_df = pf.asset_value().to_frame("Portfolio")
        main_df = pd.concat([portfolio_df, stocks_df], axis=1)
        st.table(FactorExposure(main_df, factors_df).style.format("{:.4f}").bar(color=['#ef553b','#00cc96'],align='mid', axis=1))
        st.line_chart(pd.concat([portfolio_df, factors_df], axis=1))

def run():
    # 1. display selected fund's information
    # List of funds to analyze
    fund_sources = ['dataroma.com', 'etfdb.com']
    fund_source = st.sidebar.selectbox("Select Funds' Source", fund_sources)
    st.write(f"**{fund_source}**")
    if fund_source == fund_sources[0]:
        funds_tickers = ["BRK", "MKL", "GFT", "psc", "LMM", "oaklx", "ic", "DJCO", "TGM",
                    "AM", "aq", "oc", "HC", "SAM", "PI", "DA", "BAUPOST", "FS", "GR"]
        fund_ticker = st.selectbox(f"Select fund from {fund_source}", funds_tickers)        
        try:
            fund_data = getData(fund_ticker)
        except ValueError as ve:
            st.write(f"Get {fund_source} data error: {ve}")
            return
    else:
        funds_tickers = ["QQQ", "SPY", "XLB", "XLC", 
                     "XLE", "XLF", "XLI", "XLK", 
                     "XLP", "XLRE", "XLU", "XLV", 
                     "XLY", "XHB", "SDY", "SMH", 
                     "ARKK", "ARKW", "ARKQ", 
                     "ARKF", "ARKG", "ARKX", 
                     "XME", "XPH", "KBE"]
        fund_ticker = st.selectbox(f"Select fund from {fund_source}", funds_tickers)        
        try:
            fund_data = getETFData(fund_ticker)
        except ValueError as ve:
            st.write(f"Get {fund_source}data error: {ve}")
            return
        
    # Fund positions
    df = fund_data[-1]
    st.subheader(fund_data[0])
    col1, col2, col3, col4 = st.columns(4)
    if fund_source == fund_sources[0]:
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
    else:
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
    start_date, end_date = input_dates()
    symbolsDate_dict={
                    'market':       'US',
                    'symbols':      df.iloc[0:10]['Ticker'].tolist(),
                    'weights':      df.iloc[0:10]['Portfolio (%)'].tolist(),
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
        weights = []
        for symbol in stocks_df.columns:
            weights.append(df.loc[symbol,'Portfolio (%)'])
        weights = weights / sum(weights)
        pf = get_pfByWeight(stocks_df, weights)
        st.write('----')
        st.write("**组合回报表现(Porfolio's Performance)**")
        plot_pf(pf, select=False, name=f"{fund_ticker}-Original Weights")

        # 2.1.3 calculate the factors effect of Original fund portfolio.
        st.write('----')
        st.write("**因子暴露分析(Factor Exposures)**")        

        show_FactorExposure(symbolsDate_dict, pf, stocks_df)
        # 2.1.4 Assets Clusters of Original fund portfolio.
        st.write("**资产层次聚类(Assets Clusters)**")
        with st.expander("The codependence or similarity matrix: pearson; Linkage method of hierarchical clustering: ward"):
            plot_AssetsClusters(stocks_df)
        # 2.1.5 plot RRG
        st.write("**相对轮动图策略(RRG)**")
        with st.expander("根据相对轮动图的rs_ratio、rs_momentum的值对于生成轮动策略，计算最优回报解"):
            symbol_benchmark = 'SPY'
            symbolsDate_dict['symbols'] += [symbol_benchmark]
            stocks_df = get_stocks(symbolsDate_dict,'close')
            pf = RRG_Strategy(symbol_benchmark, stocks_df)
            plot_pf(pf, bm_symbol=symbol_benchmark, bm_price=stocks_df[symbol_benchmark], select=True)
            
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
        # st.subheader(subpage)
        # bobs = cal_beststrategy(symbolsDate_dict, f"Owned by {fund_ticker} in {fund_data[1]}")
        # returns_df = pd.DataFrame()
        # bobs_df = pd.DataFrame()
        # for bob_dict in bobs:
        #     returns_df[bob_dict['symbol']] = bob_dict['pf'].value()
        #     bobs_df = bobs_df.append({'symbol': bob_dict['symbol'],
        #                             'strategy name': bob_dict['strategy name'],
        #                             'param_dict': json.dumps(bob_dict['param_dict']),
        #                             'sharpe ratio': bob_dict['sharpe ratio']
        #                             }, ignore_index=True)
        # st.write("**Step 1: Original Fund weights' Portfolio of BOBs**")
        # weights = []
        # for symbol in returns_df.columns:
        #     weights.append(df.loc[symbol,'Portfolio (%)'])
        # weights = weights / sum(weights)
        # pf = get_pfByWeight(returns_df, weights)

        # pie_df = pd.DataFrame({'Ticker':returns_df.columns, 'Weights': weights})
        # st.plotly_chart(
        #     px.pie(pie_df, values='Weights', names='Ticker', title="Max Return's Allocation")
        #     )
        # plot_pf(pf, select=False, name=f"{fund_ticker}-Top10 Stocks max sharpe -Original Weights")

        # st.write("**Step 2: Max Return Portfolio of BOBs**")
        # pf, newWeights = get_pfByMaxReturn(returns_df)
        # pie_df = pd.DataFrame({'Ticker':returns_df.columns, 'Weights': newWeights})
        # st.plotly_chart(
        #     px.pie(pie_df, values='Weights', names='Ticker', title="Max Return's Allocation")
        #     )
        # plot_pf(pf, select=False, name=f"{fund_ticker}-Top10 Stocks max sharpe -Max return Weights")

if check_password():
    run()