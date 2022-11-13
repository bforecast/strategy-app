import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime
import riskfolio as rp
import riskfolio.PlotFunctions as plf
import vectorbt as vbt

from utils.processing import AKData
from utils.vbt import get_pfByWeight,  plot_pf

def report(
        returns,
        w,
        rm="MV",
        rf=0,
        alpha=0.05,
        others=0.05,
        nrow=25,
        height=6,
        width=14,
        t_factor=252,
        ini_days=1,
        days_per_year=252,
        bins=50,
    ):
    
    cov = returns.cov()
    nav = returns.cumsum()

    fig, ax = plt.subplots(
        nrows=6,
        figsize=(width, height * 6),
        gridspec_kw={"height_ratios": [2, 1, 1.5, 1, 1, 1]},
    )

    ax[0] = plf.plot_table(
        returns,
        w,
        MAR=rf,
        alpha=alpha,
        t_factor=t_factor,
        ini_days=ini_days,
        days_per_year=days_per_year,
        ax=ax[0],
    )

    ax[2] = plf.plot_pie(
        w=w,
        title="Portfolio Composition",
        others=others,
        nrow=nrow,
        cmap="tab20",
        ax=ax[2],
    )

    ax[3] = plf.plot_risk_con(
        w=w,
        cov=cov,
        returns=returns,
        rm=rm,
        rf=rf,
        alpha=alpha,
        t_factor=t_factor,
        ax=ax[3],
    )

    ax[4] = plf.plot_hist(returns=returns, w=w, alpha=alpha, bins=bins, ax=ax[4])

    ax[[1, 5]] = plf.plot_drawdown(nav=nav, w=w, alpha=alpha, ax=ax[[1, 5]])

    year = str(datetime.now().year)

    title = "Riskfolio-Lib Report"
    subtitle = "Copyright (c) 2020-" + year + ", Dany Cajas. All rights reserved."

    fig.suptitle(title, fontsize="xx-large", y=1.011, fontweight="bold")
    ax[0].set_title(subtitle, fontsize="large", ha="center", pad=10)

    return fig

def show_OpMS(stocks_df, rm="MV", plotname=""):
    '''
    calculate portfolio Optimized max sharpe ratio
    '''
    pct_df = pd.DataFrame()
    stocks_df.dropna(axis=1,how='any', inplace=True)
    for symbol in stocks_df.columns:
        pct_df[symbol] = stocks_df[symbol].pct_change().dropna()
    port = rp.Portfolio(returns=pct_df)
    method_mu='hist'
    method_cov='hist'
    port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)
    # rm = 'MV' # Risk measure used, this time will be variance
    model="Classic"
    obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
    hist = True # Use historical scenarios for risk measures that depend on scenarios
    rf = 0 # Risk free rate
    l = 0 # Risk aversion factor, only useful when obj is 'Utility'
    weights_df = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)

    # fig = report(stocks_df, w, rm='MV', rf=0, alpha=0.05, height=6, width=14, others=0.05, nrow=25)
    # st.pyplot(fig)
   
    # Calculate returns of portfolio with optimized weights
    pfs_df=stocks_df.copy()
    pfs_df['Optimized Portfolio'] = 0
    i = 0
    
    if weights_df is None:
        st.error("No Optimzied max sharpe portfolio solution.")
        return
    for symbol, row in weights_df.iterrows():
        pfs_df['Optimized Portfolio'] += stocks_df[symbol].pct_change() * row["weights"] * 100
        i+=1
	# Display everything on Streamlit
    weights_df['Ticker'] = weights_df.index
    fig = px.pie(weights_df.iloc[0:10], values='weights', names='Ticker', title='Optimized Max Shape Portfolio Weights')
    st.plotly_chart(fig)
    # Plot Optimized Portfolio
    pf = get_pfByWeight(stocks_df, weights_df['weights'].values)
    plot_pf(pf, select=False, name=plotname)
