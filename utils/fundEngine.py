import pandas as pd
import numpy as np
import json
import datetime

import streamlit as st
import vectorbt as vbt
import akshare as ak


from utils.dataroma import *


class fundEngine(object):
    '''base class of fund engine'''
    name = "base"
    market = ""
    funds = []
    fund_name = ""
    fund_ticker = ""
    fund_update_date = ""
    fund_info = ""
    fund_period = ""
    fund_df = pd.DataFrame()

    def __init__(self):
         pass
    
    def readStocks(self, fund_ticker:str):
        return


class fe_etfdb(fundEngine):
    name = "etfdb.com"
    market = "US"
    funds = ["QQQ", "SPY", "XLB", "XLC", 
             "XLE", "XLF", "XLI", "XLK", 
             "XLP", "XLRE", "XLU", "XLV", 
             "XLY", "XHB", "SDY", "SMH", 
             "ARKK", "ARKW", "ARKQ", 
             "ARKF", "ARKG", "ARKX", 
             "XME", "XPH", "KBE"]
    funds_name = funds
    
    @vbt.cached_method
    def readStocks(self, fund_ticker:str):
        # Data Extraction
        # We obtain the HTML from the corresponding fund in Dataroma.

        html = requests.get(
            "https://etfdb.com/etf/" + fund_ticker + "/#holdings", headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:94.0) Gecko/20100101 Firefox/94.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "es-ES,es;q=0.8,en-US;q=0.5,en;q=0.3",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "same-origin",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0"
            }).content

        # Non-table Data Parsing
        soup = BeautifulSoup(html, "html.parser")
        name = soup.find('title').text
        portfolio_date = soup.find('time',class_="date-modified").text

        # Table Data Parsing
        df_list = pd.read_html(html)
        for df in df_list:
            if('Symbol Symbol' in df.columns):
                df = df.head(15)
                break
        
        # Column name corrections.
        df = df.rename(columns={"Symbol Symbol": "Ticker"})
        df = df.rename(columns={"Holding Holding": "Stock"})
        df = df.rename(columns={"% Assets % Assets": "Portfolio (%)"})

        df['Ticker'] = df['Ticker'].apply(lambda x: x.replace('.', '_'))
        df["Portfolio (%)"] = df["Portfolio (%)"].apply(lambda x: pd.to_numeric(x.split("%")[0]))
        df.index = df['Ticker']

        self.fund_ticker = fund_ticker
        self.fund_name = name
        self.fund_update_date = portfolio_date
        self.fund_period = portfolio_date
        self.fund_df = df
        return

class fe_dataroma(fundEngine):
    name = "dataroma.com"
    market = "US"
    funds = ["BRK", "MKL", "GFT", "psc", "LMM", "oaklx", "ic", "DJCO", "TGM",
                "AM", "aq", "oc", "HC", "SAM", "PI", "DA", "BAUPOST", "FS", "GR"]
    funds_name = funds
    
    def __init__(self):
        try:
            funds_df = getSuperInvestors()
            self.funds = funds_df['ticker'].tolist()
            self.funds_name = funds_df['Portfolio Manager - Firm'].tolist()
        except ValueError as ve:
            st.write(f"Get {self.name} data error: {ve}")


    @vbt.cached_method
    def readStocks(self, fund_name:str):
        self.fund_ticker = self.funds[self.funds_name.index(fund_name)]
        name, period, portfolio_date, df = getData(self.fund_ticker)
        self.fund_name = name
        self.fund_period = period
        self.fund_update_date = portfolio_date
        self.fund_df = df
        return
    
class fe_akshare(fundEngine):
    name = "天天基金网-3家5星"
    market = "CN"

    def __init__(self):
        try:
            fund_rating_all_df = ak.fund_rating_all()
            fund_rating_all_df = fund_rating_all_df[(fund_rating_all_df['5星评级家数']==3) & (fund_rating_all_df['类型']=='混合型-偏股')]
            self.funds = fund_rating_all_df['代码'].tolist()
            self.funds_name = fund_rating_all_df['简称'].tolist()
        except ValueError as ve:
            st.write(f"Get {self.name} data error: {ve}")
        
    @vbt.cached_method
    def readStocks(self, fund_name: str):
        try:
            self.fund_ticker = self.funds[self.funds_name.index(fund_name)]
            df = ak.fund_portfolio_hold_em(symbol=self.fund_ticker, date=datetime.date.today().strftime('%Y'))
            # Column name corrections.
            df = df[(df["季度"] == df['季度'][0])]
            self.fund_period = df['季度'][0][0:8]
            df = df[['股票代码', '股票名称', '占净值比例']]
            df = df.rename(columns={"股票代码": "Ticker"})
            df = df.rename(columns={"股票名称": "Stock"})
            df = df.rename(columns={"占净值比例": "Portfolio (%)"})
            df.index = df['Ticker']
            # self.fund_name = self.fund_rating_all_df.loc[self.fund_rating_all_df['代码']==fund_ticker, '简称'].values[0]
            self.fund_name = fund_name
            self.fund_update_date = datetime.date.today().strftime('%Y-%m-%d')
            self.fund_df = df
        except ValueError as ve:
            st.write(f"Get {self.name}-{self.fund_ticker} data error: {ve}")
        return
    
def get_fundSources():
    return ['dataroma.com', 'etfdb.com', '天天基金网-3家5星']

def get_fundEngine(fund_source:str):
    if(fund_source == 'dataroma.com'):
        return fe_dataroma()
    elif(fund_source == 'etfdb.com'):
        return fe_etfdb()
    else:
        return fe_akshare()
