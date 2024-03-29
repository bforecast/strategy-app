from datetime  import datetime, date, timedelta
import pytz
import numpy as np

import streamlit as st
from streamlit_quill import st_quill

from utils.portfolio import Portfolio

def input_dates(by='unique'):
    start_date = st.sidebar.date_input("Start date?", date(2018, 1, 1), key=by+'_start_date')
    end_date = st.sidebar.date_input("End date?", date.today(),  key=by+'_end_date')
    start_date = datetime(year=start_date.year, month=start_date.month, day=start_date.day, tzinfo=pytz.utc)
    end_date = datetime(year=end_date.year, month=end_date.month, day=end_date.day, tzinfo=pytz.utc)
    return start_date, end_date

def form_SavePortfolio(symbolsDate_dict, strategyname:str, strategy_param:dict, pf):
    with st.expander("Edit description and Save"):
        with st.form("form_" + strategyname):
            desc_str = st_quill(value= f"{strategyname},  Param_dict: {strategy_param}", html= True)
            submitted = st.form_submit_button("Save")
            if submitted:
                portfolio = Portfolio()
                if portfolio.add(symbolsDate_dict, strategyname, strategy_param, pf, desc_str):
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

    if st.secrets["password"]=="":
        return True
        
    if "password_correct" not in st.session_state:
        if "password" not in st.session_state:
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
    market = st.sidebar.radio("Which Market?", ('US', 'CN', 'HK'), horizontal= True)
    if market == 'US':
        symbols_string = st.sidebar.text_input("Enter all stock tickers to be included in portfolio separated by commas \
                                WITHOUT spaces, e.g. 'AMZN,NFLX,GOOG,AAPL'", '', key="textinput" + "_symbols").upper()
    elif market == 'CN':
        symbols_string = st.sidebar.text_input("Enter all stock tickers to be included in portfolio separated by commas \
                                WITHOUT spaces, e.g. '601318,000001'", '', key="textinput" + "_symbols")
    else:
        symbols_string = st.sidebar.text_input("Enter all stock tickers to be included in portfolio separated by commas \
                                WITHOUT spaces, e.g. '00700,01171'", '', key="textinput" + "_symbols")
    symbols = []
    if len(symbols_string) > 0:
        symbols = symbols_string.strip().split(',')

    start_date = st.sidebar.date_input("Start date?", date(2018, 1, 1))
    end_date = st.sidebar.date_input("End date?", date.today()- timedelta(days = 1))
    start_date = datetime(year=start_date.year, month=start_date.month, day=start_date.day, tzinfo=pytz.utc)
    end_date = datetime(year=end_date.year, month=end_date.month, day=end_date.day, tzinfo=pytz.utc)
    
    return {
            "market":   market,
            "symbols":  symbols,
            "start_date": start_date,
            "end_date": end_date,
        }

def params_selector(params):
    params_parse = dict()
    st.write("**Optimization Parameters:**")

    col1, col2 = st.columns([3, 1])
    with col1:
        params_parse['RARM'] = st.selectbox('Risk Adjusted Return Method', 
                        ['sharpe_ratio', 'annualized_return', 'deflated_sharpe_ratio', 'calmar_ratio', 'sortino_ratio', 
                         'omega_ratio', 'information_ratio', 'tail_ratio'])
    with col2:
        params_parse['WFO'] = st.selectbox("Walk Forward Optimization",
                        ['None', 'Non-anchored', 'Anchored'])

    for param in params:
        col1, col2 = st.columns([3, 1])
        with col1:
            gap = (param["max"]-param["min"]) * 0.5
            if param["step"] == 0:
                value = st.slider("Select " + param["name"], min_value= param["min"], max_value= param['max'], step= 1)
                values = [value, value]
            else:
                if param['type'] == 'int':
                    gap = int(gap)
                    bottom = max(0, param["min"] - gap)
                else:
                    bottom = max(0.0, param["min"] - gap)

                values =st.slider("Select a range of " + param["name"],
                                bottom, param['max'] + gap, (param["min"], param["max"]))
        with col2:
            step_number = st.number_input("step of " + param["name"], value=param["step"])

        if step_number == 0:
             params_parse[param["name"]] = [values[0]]
        else:
            params_parse[param["name"]] = np.arange(values[0], values[1], step_number)

    return params_parse
