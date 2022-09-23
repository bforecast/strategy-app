import pandas as pd
import psycopg2
from psycopg2.extras import Json, DictCursor
from datetime import datetime, date
import pytz
import json
import numpy as np
import os

import streamlit as st
import vectorbt as vbt
import config
import warnings

from utils.processing import AKData
from vbt_strategy import MOM, PairTrading
from utils.db import init_connection

warnings.filterwarnings('ignore')
# Initialize connection.
connection, cursor = init_connection()

class Portfolio(object):
    def __init__(self):
        self.df = pd.read_sql("SELECT * FROM portfolio", connection)
        # self.df.set_index('id', inplace=True)

    def get_df(self):
        return self.df

    def add(self, symbolsDate_dict, strategy, strategy_param, pf)->bool:
            market = symbolsDate_dict['market']
            symbols = symbolsDate_dict['symbols']
            start_date = symbolsDate_dict['start_date']
            end_date = symbolsDate_dict['end_date']

            name = strategy + '_' + '&'.join(symbols)
            filename = str(int(datetime.now().timestamp())) + '.pf'
            pf.save(config.PORTFOLIO_PATH + filename)
            total_return = round(pf.stats('total_return')[0], 2)
            sharpe_ratio = round(pf.stats('sharpe_ratio')[0], 2)
            maxdrawdown = round(pf.stats('max_dd')[0], 2)
            annual_return = round(pf.annualized_return(), 2)
            description = strategy
            
            try:
                tickers = "','".join(symbols)
                tickers = "'" + tickers + "'"
                sql_stat = f"SELECT * FROM stock WHERE symbol in ({tickers})"
                cursor.execute(sql_stat)
                stocks = cursor.fetchall()
                if len(stocks) == len(symbols):
                    param_json = json.dumps(strategy_param)
                    tickers = ','.join(symbols)

                    total_return = round(pf.stats('total_return')[0], 2)
                    sharpe_ratio = round(pf.stats('sharpe_ratio')[0], 2)
                    maxdrawdown = round(pf.stats('max_dd')[0], 2)
                    annual_return = round(pf.annualized_return(), 2)
                    
                    sql_stat = "INSERT INTO portfolio (name, description, create_date, start_date, end_date, total_return, annual_return, sharpe_ratio, maxdrawdown, filename, param_dict, strategy, symbols, market)" + \
                                f" VALUES('{name}','{description}','{datetime.today()}','{start_date}','{end_date}',{total_return},{annual_return},{sharpe_ratio},{maxdrawdown},'{filename}','{param_json}','{strategy}','{tickers}','{market}')"
                    # sql_stat = sql_stat + " RETURNING id;"
                    cursor.execute(sql_stat)
                    # strategy = cursor.fetchone()
                    connection.commit()
                else:
                    print("some of stocks are invalid.")
                    return False

            except Exception  as e:
                print("...", e)
                connection.rollback()
                return False
        
            return True

    def delete(self, id)->bool:
        try:
            sql_stat = f"DELETE FROM portfolio WHERE id= {id}"
            cursor.execute(sql_stat)
            connection.commit()
            filename = self.df.loc[self.df['id']==id, 'filename'].values[0]
            os.remove(config.PORTFOLIO_PATH + filename)
            return True
        except Exception  as e:
            print("'Fail to Delete the Portfolio...", e)
            connection.rollback()
            return False
        
    def update(self, id)->bool:
        end_date= date.today()
        end_date = datetime(year=end_date.year, month=end_date.month, day=end_date.day, tzinfo=pytz.utc)
        market = self.df.loc[self.df['id']==id, 'market'].values[0]
        symbols = self.df.loc[self.df['id']==id, 'symbols'].values[0].split(',')
        strategy = self.df.loc[self.df['id']==id, 'strategy'].values[0]
        start_date = self.df.loc[self.df['id']==id, 'start_date'].values[0]
        param_dict = self.df.loc[self.df['id']==id, 'param_dict'].values[0]
        ofilename = self.df.loc[self.df['id']==id, 'filename'].values[0]
        Data = AKData(market)
        symbol_list = symbols.copy()
        price_df = pd.DataFrame()
        if isinstance(start_date, np.datetime64):
            start_date=pd.to_datetime(start_date)

        for symbol in symbol_list:
            stock_df = Data.download(symbol, start_date, end_date)
            if stock_df.empty:
                st.warning(f"Warning: stock '{symbol}' is invalid or missing. Ignore it", icon= "⚠️")
                symbols.remove(symbol)
            else:
                price_df[symbol] = stock_df['close']
        if len(price_df) == 0:
            st.error('None stock left',  icon="🚨")
        else:
            if strategy == 'MOM':
                pf = MOM.update(price_df[symbols[0]], param_dict)
                # plot_pf(pf)
                # update_portfolio(id, end_date, pf)
                total_return = round(pf.stats('total_return')[0], 2)
                sharpe_ratio = round(pf.stats('sharpe_ratio')[0], 2)
                maxdrawdown = round(pf.stats('max_dd')[0], 2)
                annual_return = round(pf.annualized_return(), 2)
                
                try:
                    filename = str(int(datetime.now().timestamp())) + '.pf'
                    sql_stat = f"UPDATE portfolio SET end_date='{end_date}', total_return={total_return}, annual_return={annual_return}, sharpe_ratio={sharpe_ratio}, maxdrawdown={maxdrawdown}, filename='{filename}'" 
                    sql_stat = sql_stat + f" WHERE id={id};"
                    cursor.execute(sql_stat)
                    connection.commit()
                    pf.save(config.PORTFOLIO_PATH + filename)
                    os.remove(config.PORTFOLIO_PATH + ofilename)

                except Exception  as e:
                    print("...", e)
                    connection.rollback()
                    return False
            
                return True