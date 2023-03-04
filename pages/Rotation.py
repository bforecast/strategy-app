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

# RRG functions
def rs_ratio(prices_df, benchmark, window=12):
    rs_df = pd.DataFrame([], index = prices_df.index.unique())
    for series in prices_df:
        rs = (prices_df[series].divide(benchmark)) * 100
        # rs_window = rs.ewm(span=window).mean()
        rs_window = rs.rolling(window).mean()
        rs_diff = rs_window.diff()/rs_window
        rs_diff = rs_diff.rolling(window).mean()    # 顺滑
        rs_ratio = 100 + ((rs_window - rs_window.mean()) / rs_window.std())
        rs_momentum = 100 + ((rs_diff - rs_diff.mean()) / rs_diff.std())
        rs_df[series+'_rs_ratio'] = rs_ratio
        rs_df[series+'_rs_momentum'] = rs_momentum
    rs_df.dropna(axis=0, how='any', inplace=True) 
    return rs_df

annotation = [  
                dict(x=1,
                    xref='paper',  #使用相对坐标
                    y=1,
                    yref='paper',
                    text='Leading',
                    showarrow=False,  # 不显示箭头
                    font=dict(size=15, color="green")),
                dict(x=0,
                    xref='paper',  #使用相对坐标
                    y=1,
                    yref='paper',
                    text='Improving',
                    showarrow=False,  # 不显示箭头
                    font=dict(size=15, color="blue")),
                dict(x=0,
                    xref='paper',  #使用相对坐标
                    y=0,
                    yref='paper',
                    text='Lagging',
                    showarrow=False,  # 不显示箭头
                    font=dict(size=15, color="red")),
                dict(x=1,
                    xref='paper',  #使用相对坐标
                    y=0,
                    yref='paper',
                    text='Weakening',
                    showarrow=False,  # 不显示箭头
                    font=dict(size=15, color="purple")),
            ]
def plot_RRG(rs_ratio_df, symbol):
    fig = go.Figure()
    for symbol in symbols:
        fig.add_trace(go.Scatter(x=rs_ratio_df[f'{symbol}_rs_ratio'], 
                                y= rs_ratio_df[f'{symbol}_rs_momentum'], 
                                mode='lines+markers',
                                marker=dict(symbol="circle-open-dot",
                                            size=6,
                                            # angleref="previous",
                                            ),
                                name=symbol,
                            )
                    )
        fig.add_trace(go.Scatter(x=rs_ratio_df[f'{symbol}_rs_ratio'][-1:], 
                                y= rs_ratio_df[f'{symbol}_rs_momentum'][-1:], 
                                mode='markers',
                                marker=dict(
                                        size=12,
                                    ),
                                showlegend=False
                            )
                    )
    fig.update_layout(
                title= dict(text="相对轮动图RRG"), 
                xaxis_title_text = '相对比率Ratio', 
                xaxis_dtick = 1,
                xaxis_showgrid = True,
                yaxis_title_text = '相对动量Momentum', 
                yaxis_dtick = 1,
                yaxis_showgrid = True,
                # xaxis_range = [95,105],
                # yaxis_range = [95,105],
                width=1000,
                height=600,
                annotations= annotation,
            )
    draw_canvas(fig)
    st.plotly_chart(fig, use_container_width=True)

def plot_AnimateRRG(rs_ratio_df, symbol):
    # make frame figure
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    # fill in most of layout
    fig_dict["layout"]["xaxis"] = {"range": [95, 105], "title": "相对比率Ratio", "dtick": 1, "showgrid": True}
    fig_dict["layout"]["yaxis"] = {"range": [95, 105], "title": "相对动量Momentum", "dtick": 1, "showgrid": True}
    fig_dict["layout"]["hovermode"] = "closest"
    fig_dict["layout"]["width"] = 1000
    fig_dict["layout"]["height"] = 600
    fig_dict["layout"]["annotations"] = annotation
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 100, "redraw": False},
                                    "fromcurrent": True, "transition": {"duration": 300,
                                                                        "easing": "quadratic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Date:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    # make data
    for symbol in symbols:
        data_dict = {
            "x": list(rs_ratio_df[f'{symbol}_rs_ratio'][-tail_length:]),
            "y": list(rs_ratio_df[f'{symbol}_rs_momentum'][-tail_length: ]),
            "mode": "lines+markers",
            "marker": {
                'symbol': "circle-open-dot",
                'size': 6,
            },
            "line": {
                'width': 4,
            },
            "name": symbol
        }
        fig_dict["data"].append(data_dict)
        data_dict = {
            "x": list(rs_ratio_df[f'{symbol}_rs_ratio'][-1:]),
            "y": list(rs_ratio_df[f'{symbol}_rs_momentum'][-1:]),
            "mode": "markers+text",
            "marker": {
                'symbol': "circle",
                'size': 12,
            },
            "text": symbol,
            "showlegend": False
        }
        fig_dict["data"].append(data_dict)

    # make frames
    for i in range(len(rs_ratio_df)-tail_length+1):
        d = rs_ratio_df.index[i].strftime("%Y-%m-%d")
        frame = {"data": [], "name": str(d)}
        for symbol in symbols:
            data_dict = {
                "x": list(rs_ratio_df[f'{symbol}_rs_ratio'][i: i+tail_length]),
                "y": list(rs_ratio_df[f'{symbol}_rs_momentum'][i: i+tail_length]),
                "mode": "lines+markers",
                "marker": {
                    'symbol': "circle-open-dot",
                    'size': 6,
                },
                "line": {
                    'width': 4,
                },
                "name": symbol
            }
            frame["data"].append(data_dict)
            data_dict = {
                "x": list(rs_ratio_df[f'{symbol}_rs_ratio'][i+tail_length-1:i+tail_length]),
                "y": list(rs_ratio_df[f'{symbol}_rs_momentum'][i+tail_length-1:i+tail_length]),
                "mode": "markers+text",
                "marker": {
                    'symbol': "circle",
                    'size': 12,
                },
                "text": symbol,
                "showlegend": False
            }
            frame["data"].append(data_dict)

        fig_dict["frames"].append(frame)
        slider_step = {"args": [
            [d],
            {"frame": {"duration": 300, "redraw": False},
            "mode": "immediate",
            "transition": {"duration": 300}}
        ],
            "label": d,
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)

    fig_dict["layout"]["sliders"] = [sliders_dict]
    # st.write(fig_dict)

    fig = go.Figure(fig_dict)
    draw_canvas(fig)
    st.plotly_chart(fig, use_container_width=True)

def draw_canvas(fig):
    fig.add_vline(x = 100, line_color = "black", line_width=1)
    fig.add_hline(y = 100, line_color = "black", line_width=1)
    fig.add_shape({
                    'type':'rect', 
                    'x0': 95, 'x1': 100, 
                    'y0': 95, 'y1': 100, 
                }, line={
                    'width': 1
                }, fillcolor='pink', opacity=.1)
    fig.add_shape({
                    'type':'rect', 
                    'x0': 95, 'x1': 100, 
                    'y0': 100, 'y1': 105, 
                }, line={
                    'width': 1
                }, fillcolor='blue', opacity=.1)
    fig.add_shape({
                    'type':'rect', 
                    'x0': 100, 'x1': 105, 
                    'y0': 100, 'y1': 105, 
                }, line={
                    'width': 1
                }, fillcolor='green', opacity=.1)
    fig.add_shape({
                    'type':'rect', 
                    'x0': 100, 'x1': 105, 
                    'y0': 100, 'y1': 95, 
                }, line={
                    'width': 1
                }, fillcolor='yellow', opacity=.1)



# start_date, end_date = input_dates()
symbolsDate_dict = input_SymbolsDate()
if symbolsDate_dict['market'] == 'US':
    symbol_benchmark = 'SPY'    #标普500
    if len(symbolsDate_dict['symbols']) == 0:
        symbolsDate_dict['symbols'] = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
elif symbolsDate_dict['market'] == 'CN':
    symbol_benchmark = 'sh510210'   #上证指数
else:
    symbol_benchmark = '80000'  #HSI

symbols_target = symbolsDate_dict['symbols'].copy()
symbolsDate_dict['symbols'] +=  [symbol_benchmark]
stocks_df = get_stocks(symbolsDate_dict,'close')

col1, col2, col3, col4 =  st.columns([5, 1, 1, 1])
with col1:
    symbols = st.multiselect(label ='symbols', options = symbols_target, default = symbols_target)
with col2:
    period = st.selectbox('周期', ['D', 'W', 'M'], 1)
with col3:
    window = st.selectbox('Window', [1, 5, 10, 12, 20], 2)    
with col4:
    tail_length = st.slider('尾线长度', 1, 20, 6)

stocks_df = stocks_df.resample(period).ffill()
# st.write(stocks_df)
# rs_ratio_df = rs_ratio(stocks_df[symbols], stocks_df[symbol_benchmark], window)
# rm_momentum_df = rs_momentum(rs_ratio_df, window)
rs_ratio_df = rs_ratio(stocks_df[symbols], stocks_df[symbol_benchmark], window)
# st.write(rs_ratio_df)
# fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
# fig.add_trace(go.Scatter(x=stocks_df.index, y=stocks_df[symbols[0]], name=symbols[0]), row=1, col=1)
# fig.add_trace(go.Scatter(x=stocks_df.index, y=stocks_df[symbols[0]]/stocks_df[symbol_benchmark]/10, name=f'{symbols[0]}:${symbol_benchmark}'), row=2, col=1)
# fig.add_trace(go.Scatter(x=rs_ratio_df.index, y=rs_ratio_df[f'{symbols[0]}_rs_ratio'], name = 'RS-Ratio'), row=3, col=1)
# fig.add_trace(go.Scatter(x=rs_ratio_df.index, y=rs_ratio_df[f'{symbols[0]}_rs_momentum'], name = 'RS-Momentum', ), row=3, col=1)
# st.plotly_chart(fig, use_container_width=True)

# rs_ratio_df = rs_ratio_df[-tail_length:]
# rm_momentum_df = rm_momentum_df[-tail_length:]
# plot_RRG(rs_ratio_df, symbols)

plot_AnimateRRG(rs_ratio_df, symbols)