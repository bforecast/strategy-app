import numpy as np
import pandas as pd
import io

import streamlit as st
import vectorbt as vbt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vectorbt.portfolio.base import Portfolio
from .overfitting import CSCV
from utils.db import get_SymbolName

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

def plot_pf(pf, name= "", select=True, bm_symbol=None, bm_price=None):
    ## select control wheather display the subplots multiselect box
    ## bm_symbol is benchmark symbol, bm_price is benchmark's daily prices
    if len(pf.orders.records_readable) == 0:
        st.warning("No records in Portfolio.", icon= "⚠️")
        return
    # 1.initialize
    vbt.settings.array_wrapper['freq'] = 'days'
    vbt.settings.returns['year_freq'] = '252 days'
    vbt.settings.portfolio.stats['incl_unrealized'] = True
    
    #buffer to save the html contents
    buffer = io.StringIO()
    buffer.write(f"<h2>'{name}' Strategy-App Report</h2><br>")

    # 2.plot the pf
    subplots = ['cum_returns','orders', 'trade_pnl', 'drawdowns', 'cash']
    if select:
        subplots = st.multiselect("Select subplots:", Portfolio.subplots.keys(),
                    ['cum_returns','orders', 'trade_pnl', 'drawdowns', 'cash'], key='multiselect_'+name)
    if len(subplots) > 0:
        fig = pf.plot(subplots=subplots, )
        # st.plotly_chart(fig, use_container_width=True)
        if bm_symbol:
            fig.add_trace(go.Scatter(
                                    x = bm_price.index,
                                    y = bm_price.vbt.to_returns().cumsum(axis=0) + 1, 
                                    name = bm_symbol,
                                    line_color = "red"
                                    ))
        st.plotly_chart(fig, use_container_width=True)

        #save fig to the buffer
        buffer.write("<h4>Portfolio PLot</h2>")
        fig.write_html(buffer, include_plotlyjs='cdn')

    # 3.display the stats and recodes
    tab1, tab2, tab3 = st.tabs(["Return's stats", "Orders' records", "Final Positions"])
    with tab1:
        # show Return's stats
        if bm_symbol:
            st.text(f'Benchmark is {get_SymbolName(bm_symbol)}({bm_symbol})')
            st.text(pf.returns_stats(benchmark_rets=bm_price.vbt.to_returns())) 
        else:
            st.text(pf.returns_stats())
            
    with tab2:
        # show Orders' Records
        records_df = pf.orders.records_readable
        records_df['Date'] = pd.to_datetime(records_df['Timestamp']).dt.date
        records_df['Amount'] = records_df['Price'] * records_df['Size'] - records_df['Fees']
        # find the ticker's name in Columns
        def find_ticker(x):
            column = x.Column
            result = ''
            if type(column)==str:
                result = column  # 'APPL'
            elif type(column)==tuple and type(column[-1])==str:
                result = column[-1]  # (10, 5, 'APPL')
            else:
                result = name.split('_')[-1]    # name = 'APPL
            return result
        
        records_df['Ticker'] = records_df.apply(find_ticker, axis=1)
        records_df['StockName'] = records_df.apply(lambda x: get_SymbolName(x['Ticker']), axis=1)
        records_df = records_df[['Date', 'Ticker', 'StockName', 'Size', 'Price', 'Fees', 'Amount', 'Side']]
        records_df.set_index('Date', inplace=True)
        records_df.sort_index(ascending=False , inplace=True)
        st.write(records_df.style.format({'Size':'{0:,.2f}', 'Price':'{0:,.2f}', 'Fees':'{0:,.4f}', 'Amount':'{0:,.2f}'}))
        
    with tab3:
        # show Finaal Positions    
        positions_df = pf.get_positions().records_readable
        # st.write(positions_df)
        positions_df = positions_df[(positions_df['Status'] == 'Open')]
        if len(positions_df) >0:
            positions_df['Ticker'] = positions_df.apply(find_ticker, axis=1)
            positions_df['StockName'] = positions_df.apply(lambda x: get_SymbolName(x['Ticker']), axis=1)
            positions_df['EntryDate'] = pd.to_datetime(positions_df['Entry Timestamp']).dt.date
            positions_df['Value'] = positions_df['Avg Exit Price'] * positions_df['Size'] - positions_df['Entry Fees'] - positions_df['Exit Fees']
            positions_df['Return'] = positions_df['Return']*100
            positions_df = positions_df[['Ticker', 'StockName', 'EntryDate', 'Size', 'Avg Entry Price', 'PnL', 'Return', 'Value']]
            positions_df.set_index('EntryDate', inplace=True)
            st.write(positions_df.style.format({'Size':'{0:,.2f}', 'Avg Entry Price':'{0:,.2f}', 'PnL':'{0:,.4f}', 'Return':'{0:,.2f}%', 'Value':'{0:,.2f}'}))
        st.text('Cash:     {:.2f}'.format(pf.cash()[-1]))
        st.text('Total:    {:.2f}'.format(pf.value()[-1]))

    # 4. save the stats and records to the html, and download    
    buffer.write("<style>table {text-align: right;}table thead th {text-align: center;}</style>")
    buffer.write("<br><h4>Return's Statistics</h4>")
    stats = pf.returns_stats()
    df = pd.DataFrame({'Items': stats.index, 'Values': stats.values})
    df.to_html(buf=buffer, float_format='{:10.2f}'.format, index=False, border=1)
    buffer.write("<br><h4>Order's Records</h4>")
    records_df.to_html(buf=buffer, float_format='{:10.2f}'.format, border=1)
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

def plot_Histogram(pf, idxmax, idxmax_annotaiton=''):
    # st.plotly_chart(pf.total_return().vbt.histplot())
    fig = go.Figure()
    fig.add_trace(go.Histogram(x = pf.total_return()*100, name = "return", histnorm='percent'))
    fig.add_vline(x = pf[idxmax].total_return()*100, line_color = "firebrick", line_dash = 'dash', 
                annotation_text=idxmax_annotaiton, annotation_font_color='firebrick')
    fig.add_vline(x = (pf.benchmark_value()[idxmax][-1]/pf.benchmark_value()[idxmax][0]-1)*100, line_color = "grey", annotation_text="Benchmark", annotation_position="top left")
    fig.update_layout(
            title= dict(text="Histogram of Returns", x=0.5, y=0.9), 
            xaxis_title_text = 'Return', # x轴label设置
            xaxis_ticksuffix = "%",
            yaxis_title_text = 'Percentage of Count', # 默认聚合函数count
            yaxis_ticksuffix = "%",
            bargroupgap=0.1,         # 组内距离
            # margin=go.layout.Margin(l=10, r=1, b=10)
        )
    st.plotly_chart(fig, use_container_width=True)

def plot_CSCV(pf, idxmax, RARM):
    # perform CSCV algorithm
    cscv = CSCV(10, RARM)
    cscv.add_daily_returns(pf.daily_returns())
    results = cscv.estimate_overfitting(plot=False)
    pbo_test = round(results['pbo_test'] * 100, 2)

    with st.expander(f'Probability of overfitting:  {pbo_test}%'):
        fig = make_subplots(rows=2, cols=2, specs=[[{},{}], [{},{"secondary_y": True}]],
                subplot_titles=(f'Probability of overfitting: {pbo_test} %', 'Performance degradation', 'Histogram of Returns', 'Stochastic dominance'))
        # plot Logits at row=1,col=1
        fig.add_trace(go.Histogram(x = results['logits'], showlegend=False), row=1, col=1)
        fig.update_xaxes(title_text="Logits", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        # plot Performance degradation at row=1,col=2
        fig.add_trace(go.Scatter(x=results['R_n_star'], y= results['R_bar_n_star'], mode='markers', showlegend=False), row=1, col=2)
        fig.update_xaxes(title_text="IS Performance", row=1, col=2)
        fig.update_yaxes(title_text="OOS Performance", row=1, col=2)
        # plot Stochastic dominance at row=2,col=2
        fig.add_trace(go.Scatter(x=results['dom_df'].index, y= results['dom_df']['optimized_IS'], name='optimized_IS', showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=results['dom_df'].index, y= results['dom_df']['non_optimized_OOS'], name='non_optimized_OOS', showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=results['dom_df'].index, y= results['dom_df']['SD2'], name='SD2(right)', showlegend=False), secondary_y=True, row=2, col=2)
        fig.update_xaxes(title_text="Performance optimized vs non-optimized", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", secondary_y=False, row=2, col=2)
        # plot histogram of returns at row=2,col=1
        fig.add_trace(go.Histogram(x = pf.total_return()*100, showlegend=False), row=2, col=1)
        fig.add_vline(x = pf[idxmax].total_return()*100, line_color = "firebrick", line_dash = 'dash', 
                    annotation_text=f"Maximize {RARM}", annotation_font_color='firebrick', annotation_position="top left", row=2, col=1)
        fig.add_vline(x = (pf.benchmark_value()[idxmax][-1]/pf.benchmark_value()[idxmax][0]-1)*100, 
                    line_color = "grey", annotation_text="Benchmark", annotation_position="top left", row=2, col=1)
        fig.update_xaxes(title_text="Returns[%]", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)

        fig.update_layout(title= dict(text="Combinatorially Symmetric Cross-validation", x=0.5, y=0.9),
                            height=550,)
        st.plotly_chart(fig, use_container_width=True)

def display_pfbrief(pf, param_dict:dict):
    init_vbtsetting()
    lastday_return = round(pf.returns()[-1], 2)

    sharpe_ratio = round(pf.stats('sharpe_ratio')[0], 2)
    maxdrawdown = round(pf.stats('max_dd')[0]/100.0, 2)
    annual_return = pf.annualized_return()

    cols = st.columns([1, 1, 1, 1, 3])
    with cols[0]:
        st.metric('Annualized', "{0:.0%}".format(annual_return))
    with cols[1]:
        st.metric('Lastday Ret', "{0:.1%}".format(lastday_return))
    with cols[2]:
        st.metric('Sharpe Ratio', '%.2f'% sharpe_ratio)
    with cols[3]:
        st.metric('Max DD', '{0:.0%}'.format(maxdrawdown))
    with cols[4]:
        param_str = dict(filter(lambda item: item[0] not in ['RARM', 'WFO'], param_dict.items()))
        st.text_area("Parameters", value = param_str, height=2, label_visibility='collapsed',disabled=True, key=id(object()))

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
