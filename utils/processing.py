import datetime, pytz
import requests
import pandas as pd
import akshare as ak
import streamlit as st

from utils.db import load_symbol

@st.cache(allow_output_mutation=True, ttl = 864000)
def get_us_symbol() -> dict:
    assets_df = ak.stock_us_spot_em()
    symbol_dict = {}
    for index, row in assets_df.iterrows():
        symbol = row['代码'].split('.')[1]
        symbol_dict[symbol] = row['代码']
    return symbol_dict   

@st.cache(allow_output_mutation=True, ttl = 864000)
def get_cn_symbol() -> dict:
    assets_df = ak.stock_zh_a_spot_em()
    symbol_dict = {}
    for index, row in assets_df.iterrows():
        symbol = row['代码']
        symbol_dict[symbol] = symbol
    return symbol_dict

@st.cache(allow_output_mutation=True, ttl = 864000)
def get_cnindex_symbol() -> dict:
    assets_df = ak.stock_zh_index_spot()
    symbol_dict = {}
    for index, row in assets_df.iterrows():
        symbol = row['代码'][2:]
        symbol_dict[symbol] = row['代码']
    return symbol_dict  

@st.cache(allow_output_mutation=True, ttl = 864000)
def get_hk_symbol() -> dict:
    assets_df = ak.stock_hk_spot_em()
    symbol_dict = {}
    for index, row in assets_df.iterrows():
        symbol = row['代码']
        symbol_dict[symbol] = row['代码']
    return symbol_dict      

@st.cache(allow_output_mutation=True)
def get_us_stock(symbol:str, start_date:str, end_date:str) -> pd.DataFrame:
    """get us stock data

    Args:
        ak_params symbol:str, start_date:str, end_date:str

    Returns:
        pd.DataFrame: _description_
    """
    return ak.stock_us_hist(symbol=symbol, start_date=start_date, end_date=end_date, adjust="qfq")

@st.cache(allow_output_mutation=True)
def get_cn_stock(symbol:str, start_date:str, end_date:str) -> pd.DataFrame:
    """get chinese stock data

    Args:
        ak_params symbol:str, start_date:str 20170301, end_date:str

    Returns:
        pd.DataFrame: _description_
    """
    return ak.stock_zh_a_hist(symbol=symbol, start_date=start_date, end_date=end_date, adjust="qfq")

@st.cache(allow_output_mutation=True)
def get_cnindex_stock(symbol:str, start_date:str, end_date:str) -> pd.DataFrame:
    """get chinese stock data东方财富网-中国股票指数-行情数据
        symbol="399282"; 指数代码，此处不用市场标识

    Args:
        ak_params symbol:str, start_date:str 20170301, end_date:str

    Returns:
        pd.DataFrame: _description_
    """
    return ak.index_zh_a_hist(symbol=symbol, start_date=start_date, end_date=end_date, adjust="qfq")


@st.cache(allow_output_mutation=True)
def get_hk_stock(symbol:str, start_date:str, end_date:str) -> pd.DataFrame:
    """get chinese stock data

    Args:
        ak_params symbol:str, start_date:str 20170301, end_date:str

    Returns:
        pd.DataFrame: _description_
    """
    return ak.stock_hk_hist(symbol=symbol, start_date=start_date, end_date=end_date, adjust="qfq")


# @st.cache(allow_output_mutation=True, ttl = 86400)
def get_cn_index(symbol:st, start_date:str, end_date:str) -> pd.DataFrame:
    """get chinese stock data历史行情数据-东方财富

    Args:
        ak_params symbol:str, start_date:str 20170301, end_date:str

    Returns:
        pd.DataFrame: _description_
    """
    result_df = ak.stock_zh_index_daily_em(symbol=symbol)
    result_df = result_df[(result_df['date'] >= start_date) & (result_df['date'] <= end_date)]
    return result_df

@st.cache(allow_output_mutation=True)
def get_arkholdings(fund:str, end_date:str) -> pd.DataFrame:
    """get ARK fund holding companies's weight
    Args:
        ak_params symbol:str, start_date:str, end_date:str

    Returns:
        pd.DataFrame: _description_
    """

    r = requests.get(f"https://arkfunds.io/api/v2/etf/holdings?symbol={fund}&date_to={end_date}")
    data = r.json()
    holdings_df = pd.json_normalize(data, record_path=['holdings'])
    return holdings_df[['date', 'ticker', 'company', 'market_value', 'share_price', 'weight']]


@st.cache(allow_output_mutation=True, ttl = 86400)
def get_feargreed(start_date:str) -> pd.DataFrame:
    BASE_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}    
    r = requests.get("{}/{}".format(BASE_URL, start_date), headers = headers)
    data = r.json()
    fg_data = data['fear_and_greed_historical']['data'] 
    fg_df = pd.DataFrame(columns= ['date', 'fear_greed'])
    for data in fg_data:
        dt = datetime.datetime.fromtimestamp(data['x'] / 1000, tz=pytz.utc)
        # fg_df = fg_df.append({'date': dt, 'fear_greed': int(data['y'])}, ignore_index=True)
        fg_df.loc[len(fg_df.index)] = [dt, int(data['y'])]
    fg_df.set_index('date', inplace=True)
    return fg_df

class AKData(object):
    def __init__(self, market):
        self.market = market
        
    def download(self, symbol:str, start_date:datetime.datetime, end_date:datetime.datetime) -> pd.DataFrame:
        stock_df = pd.DataFrame()
        symbol_df = load_symbol(symbol)
        if len(symbol_df)==1:  #self.symbol_dict.keys():
                if self.market =="CN" and len(symbol) > 6:
                    func = 'get_cn_index'
                else:
                    func = ('get_' + self.market + '_stock').lower()

                symbol_full = symbol
                if self.market =='US':
                    symbol_full = symbol_df.at[0, 'exchange'] + '.' + symbol

                stock_df = eval(func)(symbol=symbol_full, start_date=start_date.strftime("%Y%m%d"), end_date=end_date.strftime("%Y%m%d"))
                if not stock_df.empty:
                    stock_df = stock_df.iloc[:,:6]
                    stock_df.columns = ['date', 'open', 'close', 'high', 'low','volume']
                    stock_df.index = pd.to_datetime(stock_df['date'], utc=True)
        return stock_df