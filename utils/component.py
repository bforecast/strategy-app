import os
from datetime  import datetime, date
from tracemalloc import start
import pytz
import pandas as pd

import streamlit as st
import vectorbt as vbt

from utils.portfolio import Portfolio
from utils.processing import AKData
from utils.vbt_nb import plot_pf
from vbt_strategy import MOM, PairTrade

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

# def button_SavePortfolio(market:str, symbols, strategy:str, strategy_param:dict, pf, start_date,end_date):
#     col1, col2 = st.columns([1,4])
#     with col1:
#         button_save = st.button("Save")
#     with col2:
#         if button_save:
#             portfolio = Portfolio()
#             if portfolio.add(market, symbols, strategy, strategy_param, pf, start_date, end_date):
#                 st.success("Save the portfolio sucessfully.")
#             else:
#                 st.error('Fail to save the portfolio.')

def button_SavePortfolio(symbolsDate_dict, strategy:str, strategy_param:dict, pf):
    col1, col2 = st.columns([1,4])
    with col1:
        button_save = st.button("Save")
    with col2:
        if button_save:
            portfolio = Portfolio()
            if portfolio.add(symbolsDate_dict, strategy, strategy_param, pf):
                st.success("Save the portfolio sucessfully.")
            else:
                st.error('Fail to save the portfolio.')

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
    bar= """
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
    st.markdown(bar, unsafe_allow_html=True)

def show_bar():
    bar= """
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
    st.markdown(bar, unsafe_allow_html=True)


def input_SymbolsDate() -> dict:
    """akshare params
    :return: dict
    """
    market = st.sidebar.radio("Which Market?", ['US', 'CN', 'HK'])
    if market == 'US':
        symbols_string = st.sidebar.text_input("Enter all stock tickers to be included in portfolio separated by commas \
                                WITHOUT spaces, e.g. 'AMZN,NFLX,GOOG,AAPL'", '').upper()
    elif market == 'CN':
        symbols_string = st.sidebar.text_input("Enter all stock tickers to be included in portfolio separated by commas \
                                WITHOUT spaces, e.g. '601318,000001'", '')
    else:
        symbols_string = st.sidebar.text_input("Enter all stock tickers to be included in portfolio separated by commas \
                                WITHOUT spaces, e.g. '00700,01171'", '')
    symbols = []
    if len(symbols_string) > 0:
        symbols = symbols_string.strip().split(',')

    start_date = st.sidebar.date_input("Start date?", date(2018, 1, 1))
    end_date = st.sidebar.date_input("End date?", date.today())
    start_date = datetime(year=start_date.year, month=start_date.month, day=start_date.day, tzinfo=pytz.utc)
    end_date = datetime(year=end_date.year, month=end_date.month, day=end_date.day, tzinfo=pytz.utc)
    
    return {
            "market":   market,
            "symbols":  symbols,
            "start_date": start_date,
            "end_date": end_date,
        }