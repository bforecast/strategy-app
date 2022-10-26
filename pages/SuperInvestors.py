from optparse import Values
import pandas as pd
import numpy as np

import streamlit as st
import plotly.express as px
# from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

from utils.component import  check_password, input_dates
from utils.dataroma import *
from utils.PfOptimization import show_pfOpt
from analyzePortfolio import *


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
    funds_tickers = ["MKL", "GFT", "psc", "LMM", "oaklx", "ic", "DJCO", "TGM",
                    "AM", "aq", "oc", "HC", "SAM", "PI", "DA", "BAUPOST", "FS", "GR", "BRK"]
    
    fund_ticker = st.sidebar.selectbox("Select US Funds", funds_tickers)
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
        st.write('Period:                   ', fund_data[1])
        st.write('Portfolio_date:           ', fund_data[2])
        st.write('Num_of_positions:         ', df["Stock"].count())
        st.write('Top 10 holdings weight:   ', df["Portfolio (%)"].iloc[0:10].sum())
        st.write('Total value:              ', df["Value"].sum())
        st.write('Reported Price Change (%) ', df["Reported Price Change (%)"].mean())


        st.dataframe(df)
        fig = px.pie(df.iloc[0:10], values='Portfolio (%)', names='Stock', title='Top 10 holdings')
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
        if st.checkbox('Top 10 Allocate Optimization'):
            show_pfOpt(symbolsDate_dict)
        
