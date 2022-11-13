from distutils.log import error
import pandas as pd
import warnings
import sqlite3

from functools import cache

warnings.filterwarnings('ignore')

DBNAME = "db/portfolio.db" 

@cache
def init_connection():
    try:
        connection = sqlite3.connect(DBNAME, check_same_thread=False)
        cursor = connection.cursor()
        return connection, cursor
    except Exception  as e:
        print("Connnecting Database Error:", e)
        return None, None

# Initialize connection.
connection, cursor = init_connection()

@cache
def load_symbols():
        # cursor.execute("SELECT strategy_id, symbol, exchange, name FROM strategy_stock \
        #                     JOIN stock ON stock.id = strategy_stock.stock_id")
        # symbols_df = cursor.fetchall()
        result_df = pd.read_sql("SELECT * FROM stock", connection)
        return result_df

@cache
def load_symbol(symbol:str):
        # cursor.execute("SELECT strategy_id, symbol, exchange, name FROM strategy_stock \
        #                     JOIN stock ON stock.id = strategy_stock.stock_id")
        # symbols_df = cursor.fetchall()
        try:
            result_df = pd.read_sql(f"SELECT * FROM stock WHERE symbol='{symbol}'", connection)
            return result_df

        except Exception as e:
            print(f"Connnecting Database Error: {e}")
            return None

@cache
def get_symbolname(symbol:str):
        # cursor.execute("SELECT strategy_id, symbol, exchange, name FROM strategy_stock \
        #                     JOIN stock ON stock.id = strategy_stock.stock_id")
        # symbols_df = cursor.fetchall()
        try:
            result_df = pd.read_sql(f"SELECT name FROM stock WHERE symbol='{symbol}'", connection)
            return result_df.loc[0, 'name']

        except Exception as e:
            print(f"Connnecting Database Error: {e}")
            return None