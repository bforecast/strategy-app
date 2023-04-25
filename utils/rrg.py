# simulate StockCharts RRG
# for reference: https://school.stockcharts.com/doku.php?id=chart_analysis:rrg_charts#jdk_rs-ratio

import pandas as pd
import numpy as  np
import streamlit as st
import vectorbt as vbt
from time import time
from numba import njit


import plotly.graph_objs as go
from plotly.subplots import make_subplots # creating subplots

from utils.db import get_SymbolName
from utils.vbt import init_vbtsetting


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
def plot_LastRRG(rs_ratio_df, symbols):
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

def plot_AnimateRRG(rs_ratio_df, symbols, tail_length, sweetpoint=None):
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
            "mode": "lines+markers+text",
            "marker": {
                'symbol': "circle-open-dot",
                'size': 6,
            },
            "line": {
                'width': 4,
            },
            "name": symbol,
            "hovertemplate": '<b>%{hovertext}</b>',
            "hovertext" : [d.strftime("%Y-%m-%d") for d in rs_ratio_df.index[-tail_length: ]]
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
            "name": symbol,
            "hovertemplate": '<b>%{hovertext}</b>',
            "hovertext" : [d.strftime("%Y-%m-%d") for d in rs_ratio_df.index[-1: ]],           
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
                "name": symbol,
                "hovertemplate": '<b>%{hovertext}</b>',
                "hovertext" : [d.strftime("%Y-%m-%d") for d in rs_ratio_df.index[i: i+tail_length]]                
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
                "name": symbol,
                "hovertemplate": '<b>%{hovertext}</b>',
                "hovertext" : [d.strftime("%Y-%m-%d") for d in rs_ratio_df.index[i+tail_length-1:i+tail_length]],           
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
    fig_dict["layout"]["title"] = 'RELATIVE ROTATION GRAPHS'

    # st.write(fig_dict)

    fig = go.Figure(fig_dict)
    draw_canvas(fig)
    if sweetpoint:
        fig.add_shape({
                        'type':'rect', 
                        'x0': sweetpoint[0], 'x1': 105, 
                        'y0': sweetpoint[1], 'y1': 105, 
                    }, 
                    line={'width': 1},
                    # text = 'Sweet Area',
                    fillcolor='goldenrod', opacity=.5)

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

def plot_RatioMomentum(stocks_df, rs_ratio_df, symbols, symbol_benchmark):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(x=stocks_df.index, y=stocks_df[symbols[0]], name=symbols[0]), row=1, col=1)
    fig.add_trace(go.Scatter(x=stocks_df.index, y=stocks_df[symbols[0]]/stocks_df[symbol_benchmark]/10, name=f'{symbols[0]}:${symbol_benchmark}'), row=2, col=1)
    fig.add_trace(go.Scatter(x=rs_ratio_df.index, y=rs_ratio_df[f'{symbols[0]}_rs_ratio'], name = 'RS-Ratio'), row=3, col=1)
    fig.add_trace(go.Scatter(x=rs_ratio_df.index, y=rs_ratio_df[f'{symbols[0]}_rs_momentum'], name = 'RS-Momentum', ), row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

def plot_RRG(symbol_benchmark, stocks_df):
    col1, col2, col3, col4 =  st.columns([5, 1, 1, 1])
    symbols_target = []
    for s in stocks_df.columns:
        if s != symbol_benchmark:
            symbols_target.append(s)
    with col1:
        sSel = st.multiselect("Please select symbols:", symbols_target, 
                                format_func=lambda x: x+'('+get_SymbolName(x)+')', 
                                default = symbols_target)
        # symbols = st.multiselect(label ='symbols', options = symbols_target, default = symbols_target)
    with col2:
        period = st.selectbox('周期', ['D', 'W', 'M'], 1)
    with col3:
        window = st.selectbox('Window', [1, 5, 10, 12, 20], 2)    
    with col4:
        tail_length = st.slider('尾线长度', 1, 60, 10)

    stocks_df = stocks_df.resample(period).ffill()
    rs_ratio_df = rs_ratio(stocks_df[sSel], stocks_df[symbol_benchmark], window)
    plot_AnimateRRG(rs_ratio_df, sSel, tail_length)

# @vbt.cached_method
def RRG_Strategy(symbol_benchmark, stocks_df, output_bool=False):
    stocks_df[stocks_df<0] = np.nan
    symbols_target = []
    for s in stocks_df.columns:
        if s != symbol_benchmark:
            symbols_target.append(s)
    sSel = symbols_target
    # sSel = st.multiselect("Please select symbols:", symbols_target, 
    #                         format_func=lambda x: get_SymbolName(x)+'('+x+')', 
    #                         default = symbols_target)
    # Build param grid
    rs_ratio_mins = [98, 99, 100, 101, 102]
    rs_momentum_mins = [98, 98.5, 99, 99.5, 100, 100.5, 101, 101.5, 102]
    windows = [60, 100, 150, 200, 225, 250, 275, 300]

    param_product = vbt.utils.params.create_param_product([rs_ratio_mins, rs_momentum_mins, windows])
    param_tuples = list(zip(*param_product))
    param_columns = pd.MultiIndex.from_tuples(param_tuples, names=['rs_ratio', 'rs_momentum', 'rs_window'])
    RRG_indicator = get_RRGInd().run(prices=stocks_df[sSel], bm_price=stocks_df[symbol_benchmark], ratio=rs_ratio_mins, momentum=rs_momentum_mins, window=windows, param_product=True)
    sizes = RRG_indicator.size.shift(periods=1)
    init_vbtsetting()
    pf_kwargs = dict(fees=0.001, slippage=0.001, freq='1D')
    pf = vbt.Portfolio.from_orders(
                stocks_df[sSel].vbt.tile(len(param_columns), keys=param_columns), 
                sizes, 
                size_type='targetpercent', 
                group_by=param_columns.names,  # group of two columns
                cash_sharing=True,  # share capital between columns
                **pf_kwargs,
            )
    if not isinstance(pf.total_return(), np.float64):
        idxmax = pf.total_return().idxmax()
        st.write(f"The Max Total_return is {param_columns.names}:{idxmax}")
        pf = pf[idxmax]

        if output_bool:
            rs_df = pd.DataFrame()
            for s in sSel:
                rs_df[s+'_rs_ratio'] = RRG_indicator.rs_ratio[idxmax][s]
                rs_df[s+'_rs_momentum'] = RRG_indicator.rs_momentum[idxmax][s]
        
            # plot_RatioMomentum(stocks_df, rs_df, sSel[0:1], symbol_benchmark)
            rs_df.dropna(axis=0, how='all', inplace=True)
            plot_AnimateRRG(rs_df, sSel, 6, idxmax)
    return pf

def ratio_filter(x, s):
        # if x[s+'_rs_ratio']>98 and x[s+'_rs_momentum']>100.5:
        if x[s+'_rs_momentum']>100.5:
            return 100
        else:
            return 0

@njit
def rolling_logret_zscore_nb(a, b, window):
    """Calculate the log return spread."""
    spread = np.full_like(a, np.nan, dtype=np.float_)
    spread[1:] = np.log(a[1:] / a[:-1]) - np.log(b[1:] / b[:-1])
    zscore = np.full_like(a, np.nan, dtype=np.float_)
    for i in range(a.shape[0]):
        from_i = max(0, i + 1 - window)
        to_i = i + 1
        if i < window - 1:
            continue
        spread_mean = np.mean(spread[from_i:to_i])
        spread_std = np.std(spread[from_i:to_i])
        zscore[i] = (spread[i] - spread_mean) / spread_std
    return spread, zscore

def apply_rrg_nb(prices, bm_price, ratio, momentum, window):
    rs_ratio, rs_momentum = cal_RRG(prices, bm_price, window)

    size = np.zeros(prices.shape, dtype=np.float_)
    for i in range(rs_momentum.shape[0]):
        mod = i % 5
        for j in range(rs_momentum.shape[1]):
            if mod == 0:
                size[i,j] =  100 if rs_momentum[i,j]>momentum and rs_ratio[i,j]>ratio else 0
            else:
                size[i,j] = np.nan#size[i-mod,j]
    size = np.divide(size.T, np.sum(size, axis=1)).T
    # size[np.isnan(size)] = 0

    return rs_ratio, rs_momentum, size

def get_RRGInd():
    MomInd = vbt.IndicatorFactory(
        class_name = 'RS',
        input_names = ['prices', 'bm_price'],
        param_names = ['ratio', 'momentum', 'window'],
        output_names = ['rs_ratio', 'rs_momentum', 'size']
    ).from_apply_func(apply_rrg_nb)
    
    return MomInd

@njit
def np_RollingMean(series, window_size=10):
    # Initialize an empty list to store moving averages
    moving_averages = np.full(series.shape, np.nan, dtype=np.float_)

    for i  in range(series.shape[0] - window_size + 1):
    
        # Calculate the average of current window
        window_average = np.sum(series[i:i+window_size,],axis=0) / window_size
        
        # Store the average of current
        # window in moving average list
        moving_averages[i+window_size-1,:] = window_average
    return moving_averages
    

def np_RollingZscore(series, window_size=10):
    """
    Calculates the rolling z-score of a 2D series in numpy.
    
    Args:
    series (numpy.ndarray): 2D series to calculate rolling z-score on.
    window_size (int): Size of the rolling window.
    
    Returns:
    numpy.ndarray: Rolling z-score of the input series.
    """
    
    # Calculate the rolling z-score using the mean and standard deviation
    rolling_zscore = np.full(series.shape, np.nan, dtype=np.float_)
    for i in range(window_size-1, series.shape[0]):
        window = series[i-window_size+1:i+1, :]
        window_mean = np.mean(window, axis=0, keepdims=True)
        window_std = np.std(window, axis=0, keepdims=True)
        window_zscore = (window[-1, :] - window_mean) / window_std
        rolling_zscore[i, :] = window_zscore
    
    return rolling_zscore

@st.cache_data(ttl = 86400)
def cal_RRG(prices, bm_price, window=100):
    rs = prices/bm_price * 100
    Resample = 20
    # rs = np_RollingMean(rs, Resample)
    rs_ratio = np_RollingZscore(np_RollingMean(rs, Resample), window) + 100
    rs_diff = np.full(prices.shape, np.nan, dtype=np.float_)
    # rs_diff[1:,] = np.diff(rs_ratio, axis=0)/rs_ratio[:-1,]*100
    # rs_diff = rs_diff.rolling(window).mean()    # 顺滑
    rs_diff[1:,] = np.diff(rs, axis=0)/rs[:-1,]*100
    rs_diff = np_RollingMean(rs_diff, 20)
    # rs_diff[Resample:,] = np.diff(rs, n=Resample, axis=0)/rs[:-Resample,]*100
    rs_momentum = np_RollingZscore(rs_diff, window) + 100
    return rs_ratio, rs_momentum

def calculate_momentum(prices, benchmark_price, window):
    """
    Calculates momentum for multiple tickers given their prices and a benchmark price.
    
    Args:
    prices (numpy array): array of prices for multiple tickers, shape (num_prices, num_tickers)
    benchmark_price (numpy array): array of benchmark prices, shape (num_prices,)
    window (int): number of prices to use in momentum calculation
    
    Returns:
    momentum (numpy array): array of momentum values for each ticker, shape (num_tickers,)
    """
    # calculate returns for each ticker and the benchmark
    returns = np.log(prices[1:]) - np.log(prices[:-1])
    benchmark_returns = np.log(benchmark_price[1:]) - np.log(benchmark_price[:-1])
    # calculate momentum for each ticker
    momentum = np.full_like(prices, np.nan, dtype=np.float_)
    for i in range(window-1, prices.shape[0]-1):
        momentum[i] = np.mean(returns[i-window:i], axis=0) - np.mean(benchmark_returns[i-window:i], axis=0)
    # Normalized the results
    momentum = np.divide(np.subtract(momentum, np.nanmean(momentum, axis=0)), np.nanstd(momentum, axis=0))
    return momentum

