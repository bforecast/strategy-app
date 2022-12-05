import pandas as pd
import numpy as np

import streamlit as st
import vectorbt as vbt

from utils.processing import AKData
from utils.vbt import plot_pf

class BaseStrategy(object):
    '''base strategy'''
    _name = "base"
    desc = "......"
    param_dict = {}
    param_def = {}
    stock_dfs = []
    pf_kwargs = dict(fees=0.001, slippage=0.001, freq='1D')
    pf = None
    
    def __init__(self, symbolsDate_dict:dict):
        market = symbolsDate_dict['market']
        symbols = symbolsDate_dict['symbols']
        self.start_date = symbolsDate_dict['start_date']
        self.end_date = symbolsDate_dict['end_date']
        self.datas = AKData(market)
        self.stock_dfs = []
        if 'WFO' in symbolsDate_dict.keys():
            self.param_dict['WFO'] = symbolsDate_dict['WFO']
        else:
            self.param_dict['WFO'] = False
        for symbol in symbols:
            if symbol!='':
                stock_df = self.datas.get_stock(symbol, self.start_date, self.end_date)
                if stock_df.empty:
                    st.warning(f"Warning: stock '{symbol}' is invalid or missing. Ignore it", icon= "⚠️")
                else:
                    self.stock_dfs.append((symbol, stock_df))
        
        # initialize param_dict using default param_def
        for param in self.param_def:
            if param["step"] == 0:
                self.param_dict[param["name"]] = [int((param["min"] + param['max'])/ 2)]
            else:
                self.param_dict[param["name"]] = np.arange(param["min"], param["max"], param["step"])
        #initialize vbt setting
        vbt.settings.array_wrapper['freq'] = 'days'
        vbt.settings.returns['year_freq'] = '252 days'
        vbt.settings.portfolio.stats['incl_unrealized'] = True

    def log(self, txt, dt=None, doprint=False):
        pass

    def maxSR(self, param, output_bool=False):
        self.param_dict.update(param)
        # try:
        if True:
            if self.run(output_bool, calledby='add'):
                if output_bool:
                    plot_pf(self.pf)
                return True
            else:
                return False
        # except Exception as e:
        #     print(f"{self._name}-maxSR throws exception: {e}")
        #     return False

    def update(self, param_dict:dict):
        """
            update the strategy with the param dictiorary saved in portfolio
        """
        if len(self.stock_dfs) == 0:
            return None
        else:
            for k, v in param_dict.items():
                self.param_dict[k] = [v]
            self.run(calledby='update')
            return self.pf

    def maxSR_WFO(self, price, entries, exits, to_period='y', window=1):
        '''
        Walk Foreward Optimization:
        to_period='y'   : update parameter of maxSR in to_period unit('Y')
        window=1        : 1 year 
        '''
        month_dups = price.index.to_period(to_period).duplicated()
        month_days = []
        for k, m in enumerate(month_dups):
            if not m:
                month_days.append(k)

        num_months = len(month_days)
        new_entries = np.full_like(price, False)
        new_exits = np.full_like(price, False)
        info_df = pd.DataFrame()
        for m in range(window-1, num_months-1):
            pf = vbt.Portfolio.from_signals(price[month_days[m-window+1]: month_days[m+1]], 
                                          entries[month_days[m-window+1]: month_days[m+1]], 
                                            exits[month_days[m-window+1]: month_days[m+1]], 
                                            **self.pf_kwargs)
            SRs = pf.sharpe_ratio()
            idxmax = SRs[SRs != np.inf].idxmax()
            # idxmax = pf.total_return().idxmax()
            new_entries[month_days[m+1]: ] = entries[month_days[m+1]: ][idxmax]
            new_exits[month_days[m+1]: ] = exits[month_days[m+1]: ][idxmax]
            # info_dict[str(price.index[month_days[m+1]])] = str(idxmax)
            info_df = info_df.append({'start': price.index[month_days[m+1]],
                            'param': str(idxmax),
                            'shape ratio': pf.sharpe_ratio()[idxmax]},
                            ignore_index = True)
        return new_entries, new_exits
            
