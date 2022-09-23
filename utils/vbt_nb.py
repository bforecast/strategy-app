import numpy as np
import pandas as pd
from datetime import datetime

from numba import njit
import streamlit as st
import vectorbt as vbt
from vectorbt.generic.nb import nanmean_nb
from vectorbt.portfolio.nb import order_nb, sort_call_seq_nb
from vectorbt.portfolio.enums import SizeType, Direction


from utils.processing import get_us_stock, get_us_symbol
from vectorbt.utils.colors import adjust_opacity

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

def show_pf(filename:str):
    vbt_pf = vbt.Portfolio.load(config.PORTFOLIO_PATH + filename)
    plot_pf(vbt_pf)

def plot_pf(vbt_pf):
    vbt.settings.array_wrapper['freq'] = 'days'
    vbt.settings.returns['year_freq'] = '252 days'
    vbt.settings.portfolio.stats['incl_unrealized'] = True
    st.plotly_chart(
        vbt_pf.plot(
            subplots=['cum_returns', 'orders','trade_pnl', 'drawdowns', 'underwater'],
            subplot_settings=dict(
            underwater=dict(
                    trace_kwargs=dict(
                        line=dict(color='#FF6F00'),
                        fillcolor=adjust_opacity('#FF6F00', 0.3)
                    )
                )
            )
        )   
    )
    st.text(vbt_pf.returns_stats()) 