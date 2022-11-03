import streamlit as st
from pandas_datareader.data import DataReader
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
import copy
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from io import BytesIO
import plotly.express as px

from utils.processing import AKData
from utils.plot import plot_cum_returns
	
def plot_efficient_frontier_and_max_sharpe(mu, S): 
	# Optimize portfolio for max Sharpe ratio and plot it out with efficient frontier curve
	ef = EfficientFrontier(mu, S)
	fig, ax = plt.subplots(figsize=(6,4))
	ef_max_sharpe = copy.deepcopy(ef)
	plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
	# Find the max sharpe portfolio
	ef_max_sharpe.max_sharpe(risk_free_rate=0.02)
	ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
	ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
	# Generate random portfolios
	n_samples = 1000
	w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
	rets = w.dot(ef.expected_returns)
	stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
	sharpes = rets / stds
	ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
	# Output
	ax.legend()
	return fig

def show_pfOpt(symbolsWeightDate_dict:dict):
	market = symbolsWeightDate_dict['market']
	symbols = symbolsWeightDate_dict['symbols']
	oweights = symbolsWeightDate_dict['weights']
	start_date = symbolsWeightDate_dict['start_date']
	end_date = symbolsWeightDate_dict['end_date']
	datas = AKData(market)
	stocks_df= pd.DataFrame() 
	oweight_dict = dict()
	for i in range(0, len(symbols)):
		symbol = symbols[i]
		if symbol!='':
			stock_df = datas.get_stock(symbol, start_date, end_date)
			if stock_df.empty:
				st.warning(f"Warning: stock '{symbol}' is invalid or missing. Ignore it", icon= "⚠️")
			else:
				stocks_df[symbol] = stock_df.close
				oweight_dict[symbol] = oweights[i]
		i+= 1

	# Plot Individual Stock Prices
	fig_price = px.line(stocks_df, title='Price of Individual Stocks')
 	# Plot Individual Cumulative Returns
	fig_cum_returns = plot_cum_returns(stocks_df, 'Cumulative Returns of Individual Stocks Starting with $100')
	# Calculatge and Plot Correlation Matrix between Stocks
	corr_df = stocks_df.corr().round(2)
	fig_corr = px.imshow(corr_df, text_auto=True, title = 'Correlation between Stocks')
			
	# Calculate expected returns and sample covariance matrix for portfolio optimization later
	mu = expected_returns.mean_historical_return(stocks_df)
	S = risk_models.sample_cov(stocks_df)

	# Get optimized weights
	ef = EfficientFrontier(mu, S)
	ef.max_sharpe(risk_free_rate=0.02)
	weights = ef.clean_weights()
	expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
	weights_df = pd.DataFrame.from_dict(weights, orient = 'index')
	weights_df.columns = ['weights']
	# Calculate returns of portfolio with optimized weights
	stocks_df['Optimized Portfolio'] = 0
	stocks_df['Orignial Portfolio'] = 0
	for symbol, weight in weights.items():
		stocks_df['Optimized Portfolio'] += stocks_df[symbol].pct_change() * weight * 100
		stocks_df['Orignial Portfolio'] += stocks_df[symbol].pct_change() * oweight_dict[symbol] 


	# Plot Cumulative Returns of Optimized Portfolio
	fig_cum_returns_optimized = plot_cum_returns(stocks_df[['Orignial Portfolio', 'Optimized Portfolio']], 'Cumulative Returns(%) of Optimized Portfolio')
	# Display everything on Streamlit
	st.subheader("Optimized Max Sharpe Portfolio Weights")
	st.plotly_chart(fig_cum_returns_optimized)
	weights_df['Ticker'] = weights_df.index
	fig = px.pie(weights_df.iloc[0:10], values='weights', names='Ticker', title='Optimized Max Shape Portfolio Weights')
	st.plotly_chart(fig)
		
	# st.subheader("Optimized Max Sharpe Portfolio Performance")
	# st.image(fig_efficient_frontier)
		
	st.subheader('Expected annual return: {}%'.format((expected_annual_return*100).round(2)))
	st.subheader('Annual volatility: {}%'.format((annual_volatility*100).round(2)))
	st.subheader('Sharpe Ratio: {}'.format(sharpe_ratio.round(2)))
		
	st.plotly_chart(fig_corr) # fig_corr is not a plotly chart
	st.plotly_chart(fig_price)
	st.plotly_chart(fig_cum_returns)
		
