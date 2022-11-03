from optparse import Values
import pandas as pd
import numpy as np

import streamlit as st
import plotly.express as px
import humanize
# from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

from utils.component import  check_password, input_dates
from utils.dataroma import *
# from utils.PfOptimization import show_pfOpt
from analyzePortfolio import *
from utils.riskfolio import show_pfOpt

# def show_siTable(df):
#     df = df[['ticker', 'Portfolio Manager-Firm','Portfolio value', 'No.of stocks',]]

#     gb = GridOptionsBuilder.from_dataframe(df)
#     # gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
#     gb.configure_selection(selection_mode="single")
#     # gb.configure_side_bar()
#     gridoptions = gb.build()

#     response = AgGrid(
#         df,
#         height=200,
#         gridOptions=gridoptions,
#         enable_enterprise_modules=True,
#         update_mode=GridUpdateMode.MODEL_CHANGED,
#         data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
#         fit_columns_on_grid_load=True,
#         header_checkbox_selection_filtered_only=True,
#     )



#     v = response['selected_rows']
#     if v:
#         st.write('Selected rows')
#         st.dataframe(v)
        
#     return v


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
    if True:
        fund_data = getData(fund_ticker)
        # Fund positions
        df = fund_data[-1]
        # df["Ticker"].apply(lambda x: stock_set.add(x))
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
                    .style.format({'Portfolio (%)':'{0:,.2f}', 'Reported Price Change (%)':'{0:,.3f}'} ), 
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
        st.subheader("Optimized Max Sharpe Portfolio Weights")
        rms_dict = {
                'MV': "Standard Deviation",
                'MAD': "Mean Absolute Deviation",
                'MSV': "Semi Standard Deviation",
                'FLPM': "First Lower Partial Moment (Omega Ratio)",
                'SLPM': "Second Lower Partial Moment (Sortino Ratio)",
                'CVaR': "Conditional Value at Risk",
                'EVaR': "Entropic Value at Risk",
                'WR':   " Worst Realization (Minimax)",
                'MDD': "Maximum Drawdown of uncompounded cumulative returns (Calmar Ratio)",
                'ADD': "Average Drawdown of uncompounded cumulative returns",
                'CDaR': "Conditional Drawdown at Risk of uncompounded cumulative returns",
                'EDaR': "Entropic Drawdown at Risk of uncompounded cumulative returns",
                'UCI': "Ulcer Index of uncompounded cumulative returns",
                }
        rm = st.selectbox('Select Risk Measures', rms_dict.keys(), format_func=lambda x: x+' ('+ rms_dict[x]+ ')')
        show_pfOpt(symbolsDate_dict, rm)
        
