import numpy as np
from datetime import datetime

import streamlit as st
import vectorbt as vbt

from vectorbt.portfolio.base import Portfolio

import config 

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

def show_pffromfile(filename:str):
    pf = vbt.Portfolio.load(config.PORTFOLIO_PATH + filename)
    plot_pf(pf)

def plot_pf(pf):
    vbt.settings.array_wrapper['freq'] = 'days'
    vbt.settings.returns['year_freq'] = '252 days'
    vbt.settings.portfolio.stats['incl_unrealized'] = True
    subplots = st.multiselect("Select subplots:", Portfolio.subplots.keys(),
                    ['cum_returns','orders', 'trade_pnl', 'drawdowns'], key='multiselect_'+str(pf.total_return()))
    if len(subplots) > 0:
        st.plotly_chart(
            pf.plot(
                subplots=subplots,
            )
        )
    st.text(pf.returns_stats()) 