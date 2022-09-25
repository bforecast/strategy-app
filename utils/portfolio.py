import pandas as pd
from datetime import datetime, date
import pytz
import json
import numpy as np
import os

import config
import warnings

from utils.db import init_connection


warnings.filterwarnings('ignore')
# Initialize connection.
connection, cursor = init_connection()

class Portfolio(object):
    """
    manage the database of portforlio, the pf file in directory
    """
    def __init__(self):
        self.df = pd.read_sql("SELECT * FROM portfolio", connection)

    def add(self, symbolsDate_dict, strategyname, strategy_param, pf)->bool:
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
        end_date = symbolsDate_dict['end_date']

        name = strategyname + '_' + '&'.join(symbols)
        filename = str(int(datetime.now().timestamp())) + '.pf'
        pf.save(config.PORTFOLIO_PATH + filename)
            
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
                annual_return = round(pf.annualized_return(), 2)
                description = strategyname
                    
                sql_stat = "INSERT INTO portfolio (name, description, create_date, start_date, end_date, total_return, annual_return, sharpe_ratio, maxdrawdown, filename, param_dict, strategy, symbols, market)" + \
                                f" VALUES('{name}','{description}','{datetime.today()}','{start_date}','{end_date}',{total_return},{annual_return},{sharpe_ratio},{maxdrawdown},'{filename}','{param_json}','{strategyname}','{tickers}','{market}')"
                cursor.execute(sql_stat)
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
            filename = self.df.loc[self.df['id']==id, 'filename'].values[0]
            os.remove(config.PORTFOLIO_PATH + filename)
            self.__init__()
            return True
        except Exception  as e:
            print("'Fail to Delete the Portfolio...", e)
            connection.rollback()
            return False
        
    def update(self, id)->bool:
        """
            update the result of portforlio to today
            input:
                id = the id of portforlio in db/table
            return:
                False = fail to update the db/table or save the pf file
                True  = update the library and save the pf file successfully

        """
        end_date= date.today()
        end_date = datetime(year=end_date.year, month=end_date.month, day=end_date.day, tzinfo=pytz.utc)
        market = self.df.loc[self.df['id']==id, 'market'].values[0]
        symbols = self.df.loc[self.df['id']==id, 'symbols'].values[0].split(',')
        strategyname = self.df.loc[self.df['id']==id, 'strategy'].values[0]
        start_date = self.df.loc[self.df['id']==id, 'start_date'].values[0]
        param_dict = self.df.loc[self.df['id']==id, 'param_dict'].values[0]
        ofilename = self.df.loc[self.df['id']==id, 'filename'].values[0]

        if isinstance(start_date, np.datetime64):
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
