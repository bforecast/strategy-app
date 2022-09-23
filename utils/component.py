import os
from datetime  import datetime, date
import pytz
import pandas as pd

import streamlit as st
import vectorbt as vbt

from utils.db import save_portfolio, delete_portfolio, update_portfolio
from utils.processing import AKData
from utils.vbt_nb import plot_pf
from vbt_strategy import MOM, PairTrading

import config


def input_symbols():
    market = st.sidebar.radio("Which Market?", ['US', 'CN', 'HK'])
    if market == 'US':
        symbols_string = st.sidebar.text_input("Enter all stock tickers to be included in portfolio separated by commas \
                                WITHOUT spaces, e.g. 'AMZN,NFLX,GOOG,AAPL'", '').upper()
        symbols = symbols_string.strip().split(',')
    elif market == 'CN':
        symbols_string = st.sidebar.text_input("Enter all stock tickers to be included in portfolio separated by commas \
                                WITHOUT spaces, e.g. '601318,000001'", '')
        symbols = symbols_string.strip().split(',')
    else:
        symbols_string = st.sidebar.text_input("Enter all stock tickers to be included in portfolio separated by commas \
                                WITHOUT spaces, e.g. '00700,01171'", '')
        symbols = symbols_string.strip().split(',')
    return market, symbols

def input_dates():
    start_date = st.sidebar.date_input("Start date?", date(2018, 1, 1))
    end_date = st.sidebar.date_input("End date?", date.today())
    start_date = datetime(year=start_date.year, month=start_date.month, day=start_date.day, tzinfo=pytz.utc)
    end_date = datetime(year=end_date.year, month=end_date.month, day=end_date.day, tzinfo=pytz.utc)
    return start_date, end_date

def button_SavePortfolio(market:str, symbols, strategy:str, strategy_param:dict, pf, start_date,end_date):
    col1, col2 = st.columns([1,4])
    with col1:
        button_save = st.button("Save")
    with col2:
        if button_save:
            name = strategy + '_' + '&'.join(symbols)
            filename = str(datetime.now().timestamp()) + '.pf'
            pf.save(config.PORTFOLIO_PATH + filename)
            total_return = round(pf.stats('total_return')[0], 2)
            sharpe_ratio = round(pf.stats('sharpe_ratio')[0], 2)
            maxdrawdown = round(pf.stats('max_dd')[0], 2)
            annual_return = round(pf.annualized_return(), 2)
            description = strategy
            status = save_portfolio(name, symbols, description, start_date, end_date, total_return, annual_return, sharpe_ratio, maxdrawdown, filename, strategy_param, strategy, market)
            if status:
                st.success("Save the portfolio sucessfully.")
            else:
                st.error('Fail to save the portfolio.')

def button_DeletePortfolio(id:int, filename:str):
    if delete_portfolio(id):
        os.remove(config.PORTFOLIO_PATH + filename)
        st.info('Delete the Portfolio Sucessfully.')
    else:
        st.error('Fail to Delete the Portfolio.')

def button_UpdatePortfolio(id, market, symbols, strategy, start_date, param_dict):
    end_date= date.today()
    end_date = datetime(year=end_date.year, month=end_date.month, day=end_date.day, tzinfo=pytz.utc)

    Data = AKData(market)
    symbol_list = symbols.copy()
    price_df = pd.DataFrame()

    for symbol in symbol_list:
        stock_df = Data.download(symbol, start_date, end_date)
        if stock_df.empty:
            st.warning(f"Warning: stock '{symbol}' is invalid or missing. Ignore it", icon= "⚠️")
            symbols.remove(symbol)
        else:
            price_df[symbol] = stock_df['close']
    if len(price_df) == 0:
        st.error('None stock left',  icon="🚨")
    else:
        if strategy == 'MOM':
            pf = MOM.update(price_df[symbols[0]], param_dict)
            plot_pf(pf)
            update_portfolio(id, end_date, pf)

def check_password():
    # hide_bar()
    """Returns `True` if the user had the correct password."""
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
    # st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        # show_bar()
        return True

def hide_bar():
    hide_bar= """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            visibility:hidden;
            width: 0px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            visibility:hidden;
        }
        </style>
    """
    st.markdown(hide_bar, unsafe_allow_html=True)

def show_bar():
    hide_bar= """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            visibility:visible;
            width: 0px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            visibility:visible;
        }
        </style>
    """
    st.markdown(hide_bar, unsafe_allow_html=True)