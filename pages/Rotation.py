import pandas as pd
import numpy as np
import json

import streamlit as st
import vectorbt as vbt

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots # creating subplots


from utils.component import  check_password, input_dates, input_SymbolsDate
import matplotlib.pyplot as plt

from utils.processing import get_stocks
from utils.rrg import plot_RRG, rs_ratio, RRG_Strategy
from utils.vbt import plot_pf

symbolsDate_dict = input_SymbolsDate()
with open("funds.json", 'r', encoding='UTF-8') as f:
    groups_dict = json.load(f)

symbol_benchmark = groups_dict[symbolsDate_dict['market']]['benchmark']
if len(symbolsDate_dict['symbols']) < 2:
    group_sel = st.selectbox("Please select symbols' group", groups_dict[symbolsDate_dict['market']]['group'].keys())
    symbolsDate_dict['symbols'] =  groups_dict[symbolsDate_dict['market']]['group'][group_sel]

symbolsDate_dict['symbols'] +=  [symbol_benchmark]
stocks_df = get_stocks(symbolsDate_dict,'close')
# pf = RRG_Strategy(symbol_benchmark, stocks_df)
# st.write(pf.stats())

pf = RRG_Strategy(symbol_benchmark, stocks_df)
plot_pf(pf, name= group_sel+' RRG Strategy', bm_symbol=symbol_benchmark, bm_price=stocks_df[symbol_benchmark], select=True)

