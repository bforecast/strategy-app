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

@st.cache(allow_output_mutation=True, ttl = 86400)
def get_us_stock(symbol:str, start_date:str, end_date:str) -> pd.DataFrame:
    """get us stock data

    Args:
        ak_params symbol:str, start_date:str, end_date:str

    Returns:
        pd.DataFrame: _description_
    """
    return ak.stock_us_hist(symbol=symbol, start_date=start_date, end_date=end_date, adjust="qfq")

@st.cache(allow_output_mutation=True, ttl = 86400)
def get_cn_stock(symbol:str, start_date:str, end_date:str) -> pd.DataFrame:
    """get chinese stock data

    Args:
        ak_params symbol:str, start_date:str 20170301, end_date:str

    Returns:
        pd.DataFrame: _description_
    """
    return ak.stock_zh_a_hist(symbol=symbol, start_date=start_date, end_date=end_date, adjust="qfq")

@st.cache(allow_output_mutation=True, ttl = 86400)
def get_cnindex_stock(symbol:str, start_date:str, end_date:str) -> pd.DataFrame:
    """get chinese stock data东方财富网-中国股票指数-行情数据
        symbol="399282"; 指数代码，此处不用市场标识

    Args:
        ak_params symbol:str, start_date:str 20170301, end_date:str

    Returns:
        pd.DataFrame: _description_
    """
    return ak.index_zh_a_hist(symbol=symbol, start_date=start_date, end_date=end_date, adjust="qfq")


@st.cache(allow_output_mutation=True, ttl = 86400)
def get_hk_stock(symbol:str, start_date:str, end_date:str) -> pd.DataFrame:
    """get chinese stock data

    Args:
        ak_params symbol:str, start_date:str 20170301, end_date:str

    Returns:
        pd.DataFrame: _description_
    """
    return ak.stock_hk_hist(symbol=symbol, start_date=start_date, end_date=end_date, adjust="qfq")


@st.cache(allow_output_mutation=True, ttl = 86400)
def get_cn_index(symbol:st, start_date:str, end_date:str) -> pd.DataFrame:
    """get chinese stock data历史行情数据-东方财富

    Args:
        ak_params symbol:str, start_date:str 20170301, end_date:str

    Returns:
        pd.DataFrame: _description_
    """
    result_df = ak.stock_zh_index_daily_em(symbol=symbol)
    start_date = start_date[0:4] + '-' + start_date[4:6] + '-' + start_date[6:]
    end_date = end_date[0:4] + '-' + end_date[4:6] + '-' + end_date[6:]

    result_df = result_df[(result_df['date'] >= start_date) & (result_df['date'] <= end_date)]
    return result_df

@st.cache(allow_output_mutation=True, ttl = 86400)
def get_cn_fund_etf(symbol:st, start_date:str, end_date:str) -> pd.DataFrame:
    """get chinese fund etf data新浪财经-基金行情的日频率行情数据
    Args:
        ak_params symbol:str, start_date:str 20170301, end_date:str

    Returns:
        pd.DataFrame: _description_
    """
    result_df = ak.fund_etf_hist_sina(symbol=symbol)
    result_df['date'] = result_df.date.map(lambda x: x.strftime('%Y-%m-%d'))

    start_date = start_date[0:4] + '-' + start_date[4:6] + '-' + start_date[6:]
    end_date = end_date[0:4] + '-' + end_date[4:6] + '-' + end_date[6:]

    result_df = result_df[(result_df['date'] >= start_date) & (result_df['date'] <= end_date)]
    return result_df

@st.cache(allow_output_mutation=True, ttl = 86400)
def get_cn_fundamental(symbol:st) -> pd.DataFrame:
    """get chinese stock pe data乐咕乐股-A 股个股指标: 市盈率, 市净率, 股息率

    Args:
        ak_params symbol:str
        Returns: pd.DataFrame: value
            trade_date	object	交易日
            pe	float64	市盈率
            pe_ttm	float64	市盈率TTM
            pb	float64	市净率
            ps	float64	市销率
            ps_ttm	float64	市销率TTM
            dv_ratio	float64	股息率
            dv_ttm	float64	股息率TTM
            total_mv	float64	总市值
    """
    result_df = ak.stock_a_lg_indicator(symbol= symbol)
    result_df.rename(columns={'trade_date': 'date'}, inplace=True)
    return result_df

@st.cache(allow_output_mutation=True, ttl = 86400)
def get_cn_valuation(symbol:st, indicator:str) -> pd.DataFrame:
    """get 百度股市通- A 股-财务报表-估值数据
        目标地址: https://gushitong.baidu.com/stock/ab-002044
        限量: 单次获取指定 symbol 和 indicator 的所有历史数据
    Args:
        symbol	    str	symbol="002044"; A 股代码
        indicator	str	indicator="总市值"; choice of {"总市值", "市盈率(TTM)", "市盈率(静)", "市净率", "市现率"}
    return:
        date	object	-
        value	float64	-
    """
    result_df = ak.stock_zh_valuation_baidu(symbol, indicator)
    return result_df

@st.cache(allow_output_mutation=True, ttl = 86400)
def get_hk_valuation(symbol:st, indicator:str) -> pd.DataFrame:
    """get 百度股市通- 港股-财务报表-估值数据
        目标地址: https://gushitong.baidu.com/stock/hk-06969
        限量: 单次获取指定 symbol 和 indicator 的所有历史数据
    Args:
        symbol      str symbol="02358"; 港股代码
        indicator	str	indicator="总市值"; choice of {"总市值", "市盈率(TTM)", "市盈率(静)", "市净率", "市现率"}
    return:
        date	object	-
        value	float64	-
    """
    result_df = ak.stock_hk_valuation_baidu(symbol, indicator)
    return result_df    

@st.cache(allow_output_mutation=True, ttl = 86400)
def stock_us_valuation_baidu(symbol: str = "AAPL", indicator: str = "总市值") -> pd.DataFrame:
    """
    百度股市通- 美股-财务报表-估值数据
    https://gushitong.baidu.com/stock/us-AAPL
    :param symbol: 股票代码
    :type symbol: str
    :param indicator: choice of {"总市值", "市盈率(TTM)", "市盈率(静)", "市净率", "市现率"}
    :type indicator: str
    :return: 估值数据
    :rtype: pandas.DataFrame
    """
    url = "https://finance.pae.baidu.com/selfselect/openapi"
    params = {
        "srcid": "51171",
        "code": symbol,
        "market": "us",
        "tag": f"{indicator}",
        "chart_select": "全部",
        "skip_industry": "0",
        "finClientType": "pc",
    }
    r = requests.get(url, params=params)
    data_json = r.json()
    if len(data_json["Result"]) == 0:
        temp_df = pd.DataFrame()
    else:
        temp_df = pd.DataFrame(data_json["Result"]["chartInfo"][0]["body"])
        temp_df.columns = ["date", "value"]
        temp_df["date"] = pd.to_datetime(temp_df["date"]).dt.date
        temp_df["value"] = pd.to_numeric(temp_df["value"])
    return temp_df

@st.cache(allow_output_mutation=True, ttl = 86400)
def get_us_valuation(symbol:st, indicator:str) -> pd.DataFrame:
    """get 百度股市通- 美股-财务报表-估值数据
        目标地址: https://gushitong.baidu.com/stock/us-AAPL
        限量: 单次获取指定 symbol 和 indicator 的所有历史数据
    Args:
        symbol      str symbol="AAPL"; 美股代码
        indicator	str	indicator="总市值"; choice of {"总市值", "市盈率(TTM)", "市盈率(静)", "市净率", "市现率"}
    return:
        date	object	-
        value	float64	-
    """
    result_df = stock_us_valuation_baidu(symbol, indicator)
    return result_df 


class AKData(object):
    def __init__(self, market):
        self.market = market
        
    def get_stock(self, symbol:str, start_date:datetime.datetime, end_date:datetime.datetime) -> pd.DataFrame:
        stock_df = pd.DataFrame()
        symbol_df = load_symbol(symbol)

        if len(symbol_df)==1:  #self.symbol_dict.keys():
                func = ('get_' + self.market + '_'+ symbol_df.at[0, 'category']).lower()

                symbol_full = symbol
                if self.market =='US':
                    symbol_full = symbol_df.at[0, 'exchange'] + '.' + symbol

                stock_df = eval(func)(symbol=symbol_full, start_date=start_date.strftime("%Y%m%d"), end_date=end_date.strftime("%Y%m%d"))
                if not stock_df.empty:
                    if len(stock_df.columns) <= 7:
                        stock_df.columns = ['date', 'open', 'close', 'high', 'low','volume']
                    else:    
                        stock_df.columns = ['date', 'open', 'close', 'high', 'low','volume', 'amount',
                                        'amplitude', 'changepercent', 'pricechange','turnoverratio']
                    stock_df.index = pd.to_datetime(stock_df['date'], utc=True)
        return stock_df

    def get_pettm(self, symbol:str) ->pd.DataFrame:
        stock_df = pd.DataFrame()
        symbol_df = load_symbol(symbol)

        if len(symbol_df)==1:  #self.symbol_dict.keys():
                if self.market =="CN" and len(symbol) > 6:
                    func = 'get_cn_index'
                else:
                    func = ('get_' + self.market + '_valuation').lower()

                stock_df = eval(func)(symbol=symbol, indicator='市盈率(TTM)')
                if not stock_df.empty:
                    stock_df.index = pd.to_datetime(stock_df['date'], utc=True)
                    stock_df = stock_df['value']
        return stock_df

    def get_pegttm(self, symbol:str) ->pd.DataFrame:
        stock_df = pd.DataFrame()
        symbol_df = load_symbol(symbol)

        if len(symbol_df)==1:  #self.symbol_dict.keys():
                if self.market =="CN" and len(symbol) > 6:
                    func = 'get_cn_index'
                else:
                    func = ('get_' + self.market + '_valuation').lower()

                pettm_df = eval(func)(symbol=symbol, indicator='市盈率(TTM)')
                mv_df = eval(func)(symbol=symbol, indicator='总市值')

                if not mv_df.empty and not pettm_df.empty:
                    pettm_df.index = pd.to_datetime(pettm_df['date'], utc=True)
                    mv_df.index = pd.to_datetime(mv_df['date'], utc=True)
                    stock_df = pd.DataFrame()
                    stock_df['pettm'] = pettm_df['value']
                    stock_df['mv'] = mv_df['value']
                    stock_df['earning'] = stock_df['mv']/stock_df['pettm']
                    stock_df['cagr'] = stock_df['earning'].pct_change(periods=252)
                    stock_df['pegttm'] = stock_df['pettm'] / stock_df['cagr']/100
                    stock_df = stock_df['pegttm']
        return stock_df


@st.cache(allow_output_mutation=True, ttl = 86400)
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