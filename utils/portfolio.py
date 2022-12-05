import pandas as pd
import pandas
from datetime import datetime, date, timedelta
import pytz
import json
import numpy as np
import os

import config
import warnings
import vectorbt as vbt

from utils.db import init_connection, get_symbolname

warnings.filterwarnings('ignore')
# Initialize connection.
connection, cursor = init_connection()

def selectpf_bySymbols(df, symbols:list):
        ids = set()
        for i, row in df.iterrows():
            for s in row['symbols'].split(','):
                if s in symbols:
                    ids.add(i)
        return df.loc[ids,:]

class Portfolio(object):
    """
    manage the database of portforlio, the pf file in directory
    """
    def __init__(self):
        self.df = pd.read_sql("SELECT * FROM portfolio", connection)

    def add(self, symbolsDate_dict:dict, strategyname:str, strategy_param, pf, description="desc")->bool:
        """
            add a portforlio to db/table
            input:
                symbolsDate_dict = dictonary of symbols and date
                strategy = the strategy name
                strategy_param =  the strategy parameters related to the portfolio
                pf = the vbt portfolio 
            return:
                False = fail to add too the db/table or save the pf file
                True  = add to the library and save the pf file successfully

        """

        market = symbolsDate_dict['market']
        symbols = symbolsDate_dict['symbols']
        start_date = symbolsDate_dict['start_date']
        end_date = pf.value().index[-1].strftime("%Y-%m-%d")


        name = strategyname + '_' + '&'.join(symbols)
        filename = str(datetime.now().timestamp()) + '.pf'
        pf.save(config.PORTFOLIO_PATH + filename)
        with open(config.PORTFOLIO_PATH + filename, 'rb') as pf_file:
          pf_blob = pf_file.read()
          pf_file.close()
          os.remove(config.PORTFOLIO_PATH + filename)    
          
          try:
            tickers = "','".join(symbols)
            tickers = "'" + tickers + "'"
            sql_stat = f"SELECT * FROM stock WHERE symbol in ({tickers})"
            cursor.execute(sql_stat)
            stocks = cursor.fetchall()
            if len(stocks) == len(symbols):
                param_json = json.dumps(strategy_param)
                tickers = ','.join(symbols)

                total_return = round(pf.stats('total_return')[0]/100.0, 2)
                sharpe_ratio = round(pf.stats('sharpe_ratio')[0], 2)
                maxdrawdown = round(pf.stats('max_dd')[0]/100.0, 2)
                annual_return = round(pf.annualized_return(), 2)
                lastday_return = round(pf.returns()[-1], 4)

                cursor.execute("INSERT INTO portfolio (id, name, description, create_date, start_date, end_date, total_return, annual_return, lastday_return, sharpe_ratio, maxdrawdown, param_dict, strategy, symbols, market, vbtpf) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (None, name, description, datetime.today(), start_date, end_date, total_return, annual_return, lastday_return, sharpe_ratio, maxdrawdown, param_json, strategyname, tickers, market, pf_blob))
                connection.commit()

            else:
                print("some of stocks are invalid.")
                return False

          except Exception  as e:
            print("...", e)
            connection.rollback()
            return False
        self.__init__()
        return True

    def delete(self, id)->bool:
        """
            delete one of portforlios
            input:
                id = the id of portforlio in db/table
            return:
                False = fail to delete from the db/table or the pf file
                True  = delete in the db/table and save the pf file successfully

        """
        
        try:
            sql_stat = f"DELETE FROM portfolio WHERE id= {id}"
            cursor.execute(sql_stat)
            connection.commit()

            self.__init__()
            return True
        except Exception  as e:
            print("'Fail to Delete the Portfolio...", e)
            connection.rollback()
            return False
        
    def update(self, id, force:bool = True)->bool:
        """
            update the result of portforlio to today
            input:
                id = the id of portforlio in db/table
            return:
                False = fail to update the db/table or save the pf file
                True  = update the library and save the pf file successfully

        """
        id = int(id)
        market = self.df.loc[self.df['id']==id, 'market'].values[0]
        symbols = self.df.loc[self.df['id']==id, 'symbols'].values[0].split(',')
        strategyname = self.df.loc[self.df['id']==id, 'strategy'].values[0]
        start_date = self.df.loc[self.df['id']==id, 'start_date'].values[0]
        param_dict = self.df.loc[self.df['id']==id, 'param_dict'].values[0]
        oend_date = pd.to_datetime(self.df.loc[self.df['id']==id, 'end_date'].values[0],  utc=True)

        if  market == 'US':
            end_date = datetime.now(pytz.timezone('US/Eastern')) - timedelta(hours=9, minutes=30)
        else:
            end_date= date.today()
        end_date = datetime(year=end_date.year, month=end_date.month, day=end_date.day, tzinfo=pytz.utc)

        if force is False and oend_date == end_date:
            print(f"Portfolio_update_{self.df.loc[self.df['id']==id, 'name'].values[0]}: Today has been updated already.")
            return True

        if type(param_dict) == str:
            param_dict = json.loads(param_dict)

        if isinstance(start_date, np.datetime64) or type(start_date) == str:
            start_date=pd.to_datetime(start_date)

        symbolsDate_dict = {
            "market":   market,
            "symbols":  symbols,
            "start_date": start_date,
            "end_date": end_date,
        }

        # get the strategy class according to strategy name
        strategy_cli = getattr(__import__(f"vbt_strategy"), f"{strategyname}Strategy")
        strategy = strategy_cli(symbolsDate_dict)
        pf = strategy.update(param_dict)
        if pf is None:
            return False

        end_date = pf.value().index[-1].strftime("%Y-%m-%d")
        total_return = round(pf.stats('total_return')[0]/100.0, 2)
        lastday_return = round(pf.returns()[-1], 4)

        sharpe_ratio = round(pf.stats('sharpe_ratio')[0], 2)
        maxdrawdown = round(pf.stats('max_dd')[0]/100.0, 2)
        annual_return = pf.annualized_return()
        if type(annual_return) == pandas.core.series.Series:
            annual_return = round(annual_return[0], 2)
        else:
            annual_return = round(annual_return, 2)
                
        try:
            filename = str(datetime.now().timestamp()) + '.pf'
            pf.save(config.PORTFOLIO_PATH + filename)
            with open(config.PORTFOLIO_PATH + filename, 'rb') as pf_file:
                pf_blob = pf_file.read()
                pf_file.close()
                os.remove(config.PORTFOLIO_PATH + filename)    
                cursor.execute("UPDATE portfolio SET end_date=?, total_return=?, lastday_return=?, annual_return=?, sharpe_ratio=?, maxdrawdown=?, vbtpf=? WHERE id=?",
                        (end_date, total_return,lastday_return, annual_return, sharpe_ratio, maxdrawdown, pf_blob, id))
                connection.commit()

        except FileNotFoundError as e:
            print(e)

        except Exception  as e:
            print("Update portfolio error:", e)
            connection.rollback()
            return False


        return True

    def updateAll(self)->bool:
        for i in range(len(self.df)):
            if not self.update(self.df.loc[i, 'id']):
                print(f"Fail to update portfolio('{self.df.loc[i, 'name']}')")
                continue
            else:
                print (f"Update portfolio('{self.df.loc[i,'name']}') successfully.")
        
        return True

    def check_records(self, dt:date) ->pd.DataFrame:
        '''
        Check all the portfolios which there're transations on dt:date
        '''
        result_df = pd.DataFrame()
        for i in range(len(self.df)):
          try:
            pf = vbt.Portfolio.loads(self.df.loc[i,'vbtpf'])
            records_df = pf.orders.records_readable.sort_values(by=['Timestamp'])
            records_df['date'] = records_df['Timestamp'].dt.date
            records_df = records_df[records_df['date']==dt]
            if len(records_df) > 0:
                records = []
                for index, row in records_df.iterrows():
                    symbol_str = self.df.loc[i,'symbols']
                    if type(row['Column']) == tuple and type(row['Column'][-1]) == str:
                        symbol_str = row['Column'][-1]
                    records.append(row['Side'] + ' ' + symbol_str + f"({get_symbolname(symbol_str)})")
                result_df = result_df.append({"name": self.df.loc[i,'name'],
                                "records": ', '.join(records)}, ignore_index=True)
          except ValueError as ve:
            print(f"portfolio-check_records:{self.df.loc[i,'name']} error --{ve}")
            continue
        
        return result_df

    def get_byName(self, svalue:str ='MOM_AAPL') ->pd.DataFrame:
        result_df = self.df[self.df['name']==svalue]
        return result_df

    def get_bySymbol(self, symbols:list =['AAPL']) ->pd.DataFrame:
        result_df = selectpf_bySymbols(self.df, symbols)
        return result_df