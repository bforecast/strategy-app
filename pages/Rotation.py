import pandas as pd
import numpy as np

import streamlit as st
import vectorbt as vbt

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots # creating subplots


from utils.component import  check_password, input_dates, input_SymbolsDate
import matplotlib.pyplot as plt

from utils.processing import get_stocks
from utils.rrg import plot_RRG

# ETFs DB https://etfdb.com/etfs/
groups_dict ={
    'US' : {
        'benchmark' : 'SPY', 
        'group' : {
                'SPDR': ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY'],
                'EW_Sector': ['RCD', 'RYH', 'RYT', 'RGI', 'RHS', 'RTM', 'RYF', 'RSP', 'RYE', 'RYU', 'EQAL', 'EWRE', 'QQEW', 'TLT', 'EMLC', 'EEM' ],
                # 'Feat' : ['TLT', 'EQAL', 'RYF', 'EMLC', 'EWRE', 'RTM', 'RYE', 'EEM', 'RYT', 'RYU', 'RHS'],
                'Market': ['IVV', 'IWM', 'QQQ', 'DIA'],
                'Country': ['IVV', 'MCHI', 'EWJ', 'EWH', 'EWU', 'EWG'],
            }
        },
    'CN' : {
        'benchmark' : 'sh000001',
        'group' : {
                'market': ['sh000016', 'sh000300', 'sz399006', 'sh000905', 'sh000852' ],
            }
        },
    'HK' : {
        'benchmark' : '80000',
        }
    }

symbolsDate_dict = input_SymbolsDate()
symbol_benchmark = groups_dict[symbolsDate_dict['market']]['benchmark']
if len(symbolsDate_dict['symbols']) == 0:
    group_sel = st.selectbox("Please select symbols' group", groups_dict[symbolsDate_dict['market']]['group'].keys())
    symbolsDate_dict['symbols'] =  groups_dict[symbolsDate_dict['market']]['group'][group_sel]

symbolsDate_dict['symbols'] +=  [symbol_benchmark]
stocks_df = get_stocks(symbolsDate_dict,'close')

plot_RRG(symbol_benchmark, stocks_df)
