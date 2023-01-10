import numpy as np
import pandas as pd

import streamlit as st
import vectorbt as vbt

from .base import BaseStrategy
from utils.vbt import plot_Histogram

class MAStrategy(BaseStrategy):
    '''MA strategy'''
    _name = "MA"
    desc = "双均线策略. 这是一种趋势跟踪策略, 该策略需要两条移动平均线, 分为快线和慢线. 当快线从下往上穿过慢线, 为黄金交叉, 开仓做多; 当快线从上往下穿过慢线, 为死亡交叉, 开仓做空."
    param_def = [
            {
            "name": "window",
            "type": "int",
            "min":  1,
            "max":  80,
            "step": 2   
            },
        ]

    @vbt.cached_method
    def run(self, calledby='add')->bool:
        #1. initialize the variables
        close_price = self.stock_dfs[0][1].close
        open_price = self.stock_dfs[0][1].open

        #2. calculate the indicators
        if calledby == 'add' or self.param_dict['WFO']!='None':
            window = self.param_dict['window']
            fast_ma, slow_ma = vbt.MA.run_combs(close_price, window=window, r=2, short_names=['fast', 'slow'])
        else:
            fast_windows = self.param_dict['fast_window']
            slow_windows = self.param_dict['slow_window']
            fast_ma = vbt.MA.run(close_price, fast_windows)
            slow_ma = vbt.MA.run(close_price, slow_windows)

        #3. remove all the name in param_def from param_dict
        for param in self.param_def:
            del self.param_dict[param['name']]

        #4. generate the vbt signal
        entries = fast_ma.ma_above(slow_ma)
        exits = fast_ma.ma_below(slow_ma)
        #Don't look into the future
        entries = entries.vbt.signals.fshift()
        exits = exits.vbt.signals.fshift()

        #5. Build portfolios
        if self.param_dict['WFO']!='None':
            entries, exits = self.maxRARM_WFO(close_price, entries, exits, calledby)
            pf = vbt.Portfolio.from_signals(close=close_price, open=open_price, entries=entries, exits=exits, **self.pf_kwargs)
        else:
            pf = vbt.Portfolio.from_signals(close=close_price, open=open_price, entries=entries, exits=exits, **self.pf_kwargs)
            if calledby == 'add':
                RARMs = eval(f"pf.{self.param_dict['RARM']}()")
                idxmax = RARMs[RARMs != np.inf].idxmax()
                if self.output_bool:
                    plot_Histogram(pf, idxmax, f"Maximize {self.param_dict['RARM']}")
                pf = pf[idxmax]

                self.param_dict['fast_window'] = int(idxmax[0])
                self.param_dict['slow_window'] =  int(idxmax[1])

        self.pf = pf
        return True

    # for multi symbols 
    # def run(self, output_bool=False, calledby='add')->bool:
    #     close_df = pd.DataFrame()
    #     open_df = pd.DataFrame()
    #     for i in range(len(self.stock_dfs)):
    #         # symbols.append(self.stock_dfs[i][0])
    #         symbol = self.stock_dfs[i][0]
    #         close_df[symbol] = self.stock_dfs[i][1].close
    #         open_df[symbol] = self.stock_dfs[i][1].open

    #     if calledby == 'add' or self.param_dict['WFO']:
    #         window = self.param_dict['window']
    #         fast_ma, slow_ma = vbt.MA.run_combs(close_df, window=window, r=2, short_names=['fast', 'slow'])
    #     else:
    #         fast_windows = self.param_dict['fast_window']
    #         slow_windows = self.param_dict['slow_window']
    #         fast_ma = vbt.MA.run(close_df, fast_windows)
    #         slow_ma = vbt.MA.run(close_df, slow_windows)

    #     entries = fast_ma.ma_above(slow_ma, level_name='entry')
    #     exits = fast_ma.ma_below(slow_ma, level_name='exit')
    #     #Don't look into the future
    #     entries = entries.vbt.signals.fshift()
    #     exits = exits.vbt.signals.fshift()
    #     num_symbol = len(self.stock_dfs)
    #     num_group = int(len(entries.columns) / num_symbol)
    #     group_list = []
    #     for n in range(num_group):
    #         group_list.extend([n]*num_symbol)
    #     group_by = pd.Index(group_list, name='group')

    #     if self.param_dict['WFO'] > 0:
    #         entries, exits = self.maxSR_WFO(close_df, entries, exits, 'y', group_by)
    #         pf = vbt.Portfolio.from_signals(close=close_df, open=open_df, 
    #                                         entries=entries, exits=exits,
    #                                         cash_sharing=True,
    #                                         **self.pf_kwargs)
    #         self.param_dict = {'WFO': self.param_dict['WFO']}
    #     else:
    #         pf = vbt.Portfolio.from_signals(close=close_df, open=open_df, 
    #                                         entries=entries, exits=exits,
    #                                         cash_sharing=True, group_by= group_by,
    #                                         **self.pf_kwargs)
    #         if calledby == 'add':
    #             SRs = pf.sharpe_ratio()
    #             idxmax = SRs[SRs != np.inf].idxmax()
    #             if output_bool:
    #                 plot_Histogram(pf, idxmax)
    #             pf = pf[idxmax]
    #             params_value = entries.columns[idxmax*num_symbol]
    #             self.param_dict = dict(zip(['fast_window', 'slow_window'], [int(params_value[0]), int(params_value[1])]))
    #     self.pf = pf
    #     return True
