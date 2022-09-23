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

warnings.filterwarnings('ignore')


# Uses st.experimental_singleton to only run once.
@st.experimental_singleton
def init_connection():
    try:
        connection = psycopg2.connect(**st.secrets["postgres"])
        cursor = connection.cursor(cursor_factory=DictCursor)
        return connection, cursor
    except Exception  as e:
        print("Connnecting Database Error:", e)
        return None, None

# Initialize connection.
connection, cursor = init_connection()

# @st.cache(allow_output_mutation=True, ttl = 86400)
def load_symbols():
        # cursor.execute("SELECT strategy_id, symbol, exchange, name FROM strategy_stock \
        #                     JOIN stock ON stock.id = strategy_stock.stock_id")
        # symbols_df = cursor.fetchall()
        result_df = pd.read_sql("SELECT * FROM stock", connection)
        return result_df

def load_symbol(symbol:str):
        # cursor.execute("SELECT strategy_id, symbol, exchange, name FROM strategy_stock \
        #                     JOIN stock ON stock.id = strategy_stock.stock_id")
        # symbols_df = cursor.fetchall()
        result_df = pd.read_sql(f"SELECT * FROM stock WHERE symbol='{symbol}'", connection)
        return result_df

def load_portfolio():
        result_df = pd.read_sql("SELECT * FROM portfolio", connection)
        return result_df


def save_portfolio(name, stock_list, description, start_date, end_date, total_return:float, 
                annual_return:float, sharpe_ratio:float, maxdrawdown:float, filename:str, strategy_param:dict, strategy:str, market:str)->bool:
        try:
            tickers = "','".join(stock_list)
            tickers = "'" + tickers + "'"
            sql_stat = f"SELECT * FROM stock WHERE symbol in ({tickers})"
            cursor.execute(sql_stat)
            stocks = cursor.fetchall()
            if len(stocks) == len(stock_list):
                param_json = json.dumps(strategy_param)
                tickers = ','.join(stock_list)
                sql_stat = "INSERT INTO portfolio (name, description, create_date, start_date, end_date, total_return, annual_return, sharpe_ratio, maxdrawdown, filename, param_dict, strategy, symbols, market)" + \
                            f" VALUES('{name}','{description}','{datetime.today()}','{start_date}','{end_date}',{total_return},{annual_return},{sharpe_ratio},{maxdrawdown},'{filename}','{param_json}','{strategy}','{tickers}','{market}')"
                sql_stat = sql_stat + " RETURNING id;"
                cursor.execute(sql_stat)
                strategy = cursor.fetchone()
                connection.commit()
            else:
                print("some of stocks are invalid.")
                return False

        except Exception  as e:
            print("...", e)
            connection.rollback()
            return False
    
        return True

def delete_portfolio(id)->bool:
    try:
        # sql_stat = f"DELETE FROM portfolio_stock WHERE portfolio_id= {id}"
        # cursor.execute(sql_stat)
        sql_stat = f"DELETE FROM portfolio WHERE id= {id}"
        cursor.execute(sql_stat)
        connection.commit()
        return True
    except Exception  as e:
        print("...", e)
        connection.rollback()
        return False
    
def update_portfolio(id, end_date, pf)->bool:
        total_return = round(pf.stats('total_return')[0], 2)
        sharpe_ratio = round(pf.stats('sharpe_ratio')[0], 2)
        maxdrawdown = round(pf.stats('max_dd')[0], 2)
        annual_return = round(pf.annualized_return(), 2)
        
        try:
            filename = str(int(datetime.now().timestamp())) + '.pf'
            pf.save(config.PORTFOLIO_PATH + filename)
            sql_stat = f"UPDATE portfolio SET end_date='{end_date}', total_return={total_return}, annual_return={annual_return}, sharpe_ratio={sharpe_ratio}, maxdrawdown={maxdrawdown}, filename='{filename}'" 
            sql_stat = sql_stat + f" WHERE id={id};"
            cursor.execute(sql_stat)
            connection.commit()
        except Exception  as e:
            print("...", e)
            connection.rollback()
            return False
    
        return True

