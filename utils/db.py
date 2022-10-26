from distutils.log import error
import pandas as pd
# import psycopg2
# from psycopg2.extras import DictCursor
import warnings
import sqlite3

import streamlit as st

warnings.filterwarnings('ignore')


# Uses st.experimental_singleton to only run once.
@st.experimental_singleton
def init_connection():
    try:
        if st.secrets['database'] == 'sqlite3':
            connection = sqlite3.connect(st.secrets['sqlite3']['dbname'], check_same_thread=False)
            cursor = connection.cursor()
        # elif st.secrets['database'] == 'postgres':
        #     connection = psycopg2.connect(**st.secrets["postgres"])
        #     cursor = connection.cursor(cursor_factory=DictCursor)
        else:
            raise error
        return connection, cursor
    except Exception  as e:
        print("Connnecting Database Error:", e)
        return None, None

# Initialize connection.
connection, cursor = init_connection()

@st.cache(allow_output_mutation=True, ttl = 86400)
def load_symbols():
        # cursor.execute("SELECT strategy_id, symbol, exchange, name FROM strategy_stock \
        #                     JOIN stock ON stock.id = strategy_stock.stock_id")
        # symbols_df = cursor.fetchall()
        result_df = pd.read_sql("SELECT * FROM stock", connection)
        return result_df

@st.cache(allow_output_mutation=True, ttl = 86400)
def load_symbol(symbol:str):
        # cursor.execute("SELECT strategy_id, symbol, exchange, name FROM strategy_stock \
        #                     JOIN stock ON stock.id = strategy_stock.stock_id")
        # symbols_df = cursor.fetchall()
        try:
            result_df = pd.read_sql(f"SELECT * FROM stock WHERE symbol='{symbol}'", connection)
            return result_df

        except Exception as e:
            st.error(f"Connnecting Database Error: {e}")
            return None

@st.cache(allow_output_mutation=True, ttl = 86400)
def get_symbolname(symbol:str):
        # cursor.execute("SELECT strategy_id, symbol, exchange, name FROM strategy_stock \
        #                     JOIN stock ON stock.id = strategy_stock.stock_id")
        # symbols_df = cursor.fetchall()
        try:
            result_df = pd.read_sql(f"SELECT name FROM stock WHERE symbol='{symbol}'", connection)
            return result_df.loc[0, 'name']

        except Exception as e:
            st.error(f"Connnecting Database Error: {e}")
            return None