import config
import sqlite3
import akshare as ak
import pandas as pd

connection = sqlite3.connect('db/portfolio.db')
cursor = connection.cursor()

def gen_us_symbol():
    # assets = api.list_assets()
    assets_df = ak.stock_us_spot_em()

    for index, row in assets_df.iterrows():
        symbol = row['代码'].split('.')[1]
        exchange = row['代码'].split('.')[0]

        cursor.execute("""
            INSERT INTO stock (name, symbol, exchange, is_etf) 
            VALUES (%s, %s, %s, false)
        """, (row['名称'], symbol, exchange))
    print(assets_df)

def gen_cn_symbol():
    assets_df = ak.stock_zh_a_spot_em()

    for index, row in assets_df.iterrows():
        symbol = row['代码']
        exchange = 'A'

        cursor.execute("""
            INSERT INTO stock (name, symbol, exchange, is_etf) 
            VALUES (%s, %s, %s, false)
        """, (row['名称'], symbol, exchange))
    print(assets_df)

def gen_cn_symbol():
    assets_df = ak.stock_zh_a_spot_em()

    for index, row in assets_df.iterrows():
        symbol = row['代码']
        exchange = 'A'

        cursor.execute("""
            INSERT INTO stock (name, symbol, exchange, is_etf) 
            VALUES (%s, %s, %s, false)
        """, (row['名称'], symbol, exchange))
    print(assets_df)

def gen_hk_symbol():
    assets_df = ak.stock_hk_spot_em()

    for index, row in assets_df.iterrows():
        symbol = row['代码']
        exchange = 'HK'

        cursor.execute("""
            INSERT INTO stock (name, symbol, exchange, is_etf) 
            VALUES (%s, %s, %s, false)
        """, (row['名称'], symbol, exchange))
    print(assets_df)


def gen_cnindex_symbol():
    assets_df = ak.stock_zh_index_spot()

    for index, row in assets_df.iterrows():
        symbol = row['代码']
        exchange = 'CNINDEX'

        cursor.execute("""
            INSERT INTO stock (name, symbol, exchange, is_etf) 
            VALUES (%s, %s, %s, false)
        """, (row['名称'], symbol, exchange))
    print(assets_df)

def gen_cnfund_etf():
    fund_etf_lof_df = ak.fund_etf_category_sina(symbol="LOF基金")
    fund_etf_etf_df = ak.fund_etf_category_sina(symbol="ETF基金")
    fund_etf_fb_df = ak.fund_etf_category_sina(symbol="封闭式基金")

    fund_eft_df=pd.concat([fund_etf_lof_df,fund_etf_etf_df,fund_etf_fb_df])
    for index, row in fund_eft_df.iterrows():
        cursor.execute(" INSERT INTO stock (id, name, symbol, exchange, is_etf, category) \
            VALUES (?, ?, ?, 'CN', false, 'FUND_ETF' )", (None, row['名称'], row['代码']))
    print(fund_eft_df)


gen_us_symbol()
gen_cn_symbol()
gen_hk_symbol()
gen_cnindex_symbol()
gen_cnfund_etf()
connection.commit()
