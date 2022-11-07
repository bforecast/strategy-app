import numpy as np
import pandas as pd
import io

import streamlit as st
import vectorbt as vbt
import plotly.express as px

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

def plot_pf(pf, select=True):
    vbt.settings.array_wrapper['freq'] = 'days'
    vbt.settings.returns['year_freq'] = '252 days'
    vbt.settings.portfolio.stats['incl_unrealized'] = True
    subplots = ['cum_returns','orders', 'trade_pnl', 'drawdowns']
    buffer = None
    if select:
        subplots = st.multiselect("Select subplots:", Portfolio.subplots.keys(),
                    ['cum_returns','orders', 'trade_pnl', 'drawdowns'], key='multiselect_'+str(pf.total_return()))
    if len(subplots) > 0:
        fig = pf.plot(subplots=subplots, )
        st.plotly_chart(fig)

        # Create an in-memory buffer
        buffer = io.BytesIO()
        # Save the figure as a pdf to the buffer
        fig.write_image(file=buffer, format="pdf")
        # Download the pdf from the buffer
        st.download_button(
            label="Download",
            data=buffer,
            file_name="portfolio.pdf",
            mime="application/pdf",
        )
        
    tab1, tab2 = st.tabs(["Return's stats", "Orders' records"])
    with tab1:
        st.text(pf.returns_stats()) 
    with tab2:
        st.text(pf.orders.records_readable.sort_values(by=['Timestamp'])) 

def plot_cum_returns(df, title):
    df = df.cumsum()
	# df.fillna(method='ffill', inplace=True)
	# fig = px.line(df, title=title)
	# return fig
    st.line_chart(df)

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

    # print(_prices.columns)

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

    # Plot annualized return against volatility, color by sharpe ratio
    annualized_return = pfs.annualized_return()
    annualized_return.index = pfs.annualized_volatility()
    st.plotly_chart(annualized_return.vbt.scatterplot(
                    trace_kwargs=dict(
                        mode='markers', 
                        marker=dict(
                            color=pfs.sharpe_ratio(),
                            colorbar=dict(
                                title='sharpe_ratio'
                            ),
                            size=5,
                            opacity=0.7
                        )
                    ),
                    xaxis_title='annualized_volatility',
                    yaxis_title='annualized_return'
                )
            )
    idxmax = (pfs.total_return().idxmax())
    pf = pfs[idxmax]
    return pf