import pandas as pd
import psycopg2
import psycopg2.extras
from datetime import datetime
import pytz
import vectorbt as vbt

import config
from utils.db import load_strategy, load_symbols, save_strategy

symbols = ['META', 'AMZN', 'NFLX', 'GOOG', 'AAPL']
start_date = datetime(2017, 1, 1, tzinfo=pytz.utc)
end_date = datetime(2020, 1, 1, tzinfo=pytz.utc)

vbt.settings.array_wrapper['freq'] = 'days'
vbt.settings.returns['year_freq'] = '252 days'
vbt.settings.portfolio['seed'] = 42
vbt.settings.portfolio.stats['incl_unrealized'] = True

pf_load = vbt.Portfolio.load('Portfolio/randomWeights_pf')
print(pf_load.stats())
save_strategy('Random',symbols , "RandomWeight", start_date, end_date, pf_load.stats('total_return')[0], 1, pf_load.stats('sharpe_ratio')[0], pf_load.stats('max_dd')[0], 'randomWeights_pf')