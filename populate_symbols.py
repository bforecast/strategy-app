import config
import alpaca_trade_api as tradeapi
import psycopg2
import psycopg2.extras
import akshare as ak

connection = psycopg2.connect(host=config.DB_HOST, port=config.DB_PORT, database=config.DB_NAME, user=config.DB_USER, password=config.DB_PASS)
cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)

# api = tradeapi.REST(config.API_KEY, config.API_SECRET, base_url=config.API_URL)
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

gen_us_symbol()
gen_cn_symbol()
gen_hk_symbol()
gen_cnindex_symbol()
connection.commit()
