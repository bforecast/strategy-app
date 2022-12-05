import numpy as np
import pandas as pd
import io

import streamlit as st
import vectorbt as vbt
import plotly.graph_objects as go

from vectorbt.portfolio.base import Portfolio

def plot_allocation(rb_pf, symbols):
    # Plot weights development of the portfolio
    rb_asset_value = rb_pf.asset_value(group_by=False)
    rb_value = rb_pf.value()
    rb_idxs = np.flatnonzero((rb_pf.asset_flow() != 0).any(axis=1))
    rb_dates = rb_pf.wrapper.index[rb_idxs]
    fig = (rb_asset_value.vbt / rb_value).vbt.plot(
        trace_names=symbols,
        trace_kwargs=dict(
            stackgroup='one'
        )
    )
    for rb_date in rb_dates:
        fig.add_shape(
            dict(
                xref='x',
                yref='paper',
                x0=rb_date,
                x1=rb_date,
                y0=0,
                y1=1,
                line_color=fig.layout.template.layout.plot_bgcolor
            )
        )
    return fig

def show_pffromfile(vbtpf):
    pf = vbt.Portfolio.loads(vbtpf)
    plot_pf(pf)

def plot_pf(pf, name= "", select=True):
    # 1.initialize
    vbt.settings.array_wrapper['freq'] = 'days'
    vbt.settings.returns['year_freq'] = '252 days'
    vbt.settings.portfolio.stats['incl_unrealized'] = True
    
    #buffer to save the html contents
    buffer = io.StringIO()
    buffer.write(f"<h2>'{name}' Strategy-App Report</h2><br>")

    # 2.plot the pf
    subplots = ['cum_returns','orders', 'trade_pnl', 'drawdowns']
    if select:
        subplots = st.multiselect("Select subplots:", Portfolio.subplots.keys(),
                    ['cum_returns','orders', 'trade_pnl', 'drawdowns'], key='multiselect_'+str(pf.total_return()))
    if len(subplots) > 0:
        fig = pf.plot(subplots=subplots, )
        st.plotly_chart(fig)
        #save fig to the buffer
        buffer.write("<h4>Portfolio PLot</h2>")
        fig.write_html(buffer, include_plotlyjs='cdn')

    # 3.display the stats and recodes
    tab1, tab2 = st.tabs(["Return's stats", "Orders' records"])
    with tab1:
        st.text(pf.returns_stats()) 
    with tab2:
        st.text(pf.orders.records_readable.sort_values(by=['Timestamp'])) 

    # 4. save the stats and records to the html, and download    
    buffer.write("<style>table {text-align: right;}table thead th {text-align: center;}</style>")
    buffer.write("<br><h4>Return's Statistics</h4>")
    stats = pf.returns_stats()
    df = pd.DataFrame({'Items': stats.index, 'Values': stats.values})
    df.to_html(buf=buffer, float_format='{:10.2f}'.format, index=False, border=1)
    buffer.write("<br><h4>Order's Records</h4>")
    pf.orders.records_readable.sort_values(by=['Timestamp']).to_html(buf=buffer, index=False, float_format='{:10.2f}'.format, border=1)
    buffer.write("<br><footer>Copyright (c) 2022 Brilliant Forecast Ltd. All rights reserved.</footer>")

    html_bytes = buffer.getvalue().encode()
    st.download_button(
            label='Download Report',
            data=html_bytes,
            file_name=f'{name}-Report.html',
            mime='text/html'
        )

def plot_cum_returns(df, title):
    df = df.cumsum()
	# df.fillna(method='ffill', inplace=True)
	# fig = px.line(df, title=title)
	# return fig
    st.line_chart(df)

def plot_Histogram(close_price, pf, idxmax):
    # st.plotly_chart(pf.total_return().vbt.histplot())
    fig = go.Figure()
    fig.add_trace(go.Histogram(x = pf.total_return()*100, name = "return", histnorm='percent'))
    fig.add_vline(x = pf[idxmax].total_return()*100, line_color = "firebrick", line_dash = 'dash', 
                annotation_text="Max Sharpe Ratio", annotation_font_color='firebrick')
    fig.add_vline(x = (close_price[-1]/close_price[0]-1)*100, line_color = "grey", annotation_text="Benchmark", annotation_position="top left")
    fig.update_layout(
            title= dict(text="Histogram of Returns", x=0.5, y=0.9), 
            xaxis_title_text = 'Return', # x轴label设置
            xaxis_ticksuffix = "%",
            yaxis_title_text = 'Percentage of Count', # 默认聚合函数count
            yaxis_ticksuffix = "%",
            bargroupgap=0.1,         # 组内距离
            margin=go.layout.Margin(l=10, r=1, b=10)
        )
    st.plotly_chart(fig)

def display_pfbrief(strategy):
    pf = strategy.pf
    param_dict = strategy.param_dict
    total_return = round(pf.stats('total_return')[0]/100.0, 2)
    lastday_return = round(pf.returns()[-1], 2)

    sharpe_ratio = round(pf.stats('sharpe_ratio')[0], 2)
    maxdrawdown = round(pf.stats('max_dd')[0]/100.0, 2)
    annual_return = pf.annualized_return()

    cols = st.columns(4 + len(param_dict))
    with cols[0]:
        st.metric('Annualized', "{0:.0%}".format(annual_return))
    with cols[1]:
        st.metric('Lastday Return', "{0:.1%}".format(lastday_return))
    with cols[2]:
        st.metric('Sharpe Ratio', '%.2f'% sharpe_ratio)
    with cols[3]:
        st.metric('Max DD', '{0:.0%}'.format(maxdrawdown))
    i = 4
    for k, v in param_dict.items():
        with cols[i]:
            st.metric(k, v)
        i = i + 1

def init_vbtsetting():
    #initialize vbt setting
    vbt.settings.array_wrapper['freq'] = 'days'
    vbt.settings.returns['year_freq'] = '252 days'
    vbt.settings.portfolio.stats['incl_unrealized'] = True


def get_pfByWeight(prices, weights):
    init_vbtsetting()
    size = pd.DataFrame.vbt.empty_like(prices, fill_value=np.nan)
    size.iloc[0] =  weights  # starting weights
    pf_kwargs = dict(fees=0.001, slippage=0.001, freq='1D')

    pf = vbt.Portfolio.from_orders(
            prices, 
            size, 
            size_type='targetpercent', 
            group_by=True,  # group of two columns
            cash_sharing=True,  # share capital between columns
            **pf_kwargs,
        )
    return pf

def get_pfByMaxReturn(prices):
    init_vbtsetting()
    num_tests = 2000

    # Generate random weights, n times
    np.random.seed(42)
    weights = []
    for i in range(num_tests):
        w = np.random.random_sample(len(prices.columns))
        w = w / np.sum(w)
        weights.append(w)

    # Build column hierarchy such that one weight corresponds to one price series
    _prices = prices.vbt.tile(num_tests, keys=pd.Index(np.arange(num_tests), name='symbol_group'))
    _prices = _prices.vbt.stack_index(pd.Index(np.concatenate(weights), name='weights'))

    # Define order size
    sizes = np.full_like(_prices, np.nan)
    sizes[0, :] = np.concatenate(weights)  # allocate at first timestamp, do nothing afterwards

    # Run simulation
    pfs = vbt.Portfolio.from_orders(
            close=_prices,
            size=sizes,
            size_type='targetpercent',
            group_by='symbol_group',
            cash_sharing=True
        ) # all weights sum to 1, no shorting, and 100% investment in risky assets

    idxmax = (pfs.total_return().idxmax())
    return pfs[idxmax], weights[idxmax]
