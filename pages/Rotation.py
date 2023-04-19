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
from utils.rrg import plot_RRG, rs_ratio, RRG_Strategy
from utils.vbt import plot_pf

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
                "iShares 5 factors" : ['MTUM', 'QUAL', 'SIZE', 'USMV', 'VLUE'],
            }
        },
    'CN' : {
        'benchmark' : 'sh000001',
        'group' : {
                'market': ['sh000016', 'sh000300', 'sz399006', 'sh000905', 'sh000852'],
                '一级行业':['sh510880', 'sh510300', 'sh510050', 'sh515030', 'sh518880', 'sh510220',
                    'sz159915', 'sh510180', 'sh510170', 'sz159949', 'sh510630', 'sh515050', 
                    'sh512660', 'sh510710', 'sz159920', 'sh515650', 'sh512480', 'sh512660', 
                    'sh515650'],
                '11行业': ['sh512690',#酒
                        'sh512010',#药
                        'sh516660',#新能源
                        'sh515220',#煤炭
                        'sh516950',#基建
                        'sh512660',#军工
                        'sh512880',#证券
                        'sz159870',#化工
                        'sh512800',#银行
                        'sh512200',#地产
                        'sh512400',#有色
                        ]
                }
        },
    'HK' : {
        'benchmark' : '80000',
        }
    }

symbolsDate_dict = input_SymbolsDate()
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

