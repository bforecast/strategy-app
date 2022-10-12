import numpy as np
import pandas as pd
import talib
from itertools import combinations

import streamlit as st
import vectorbt as vbt

from .base import BaseStrategy

from numba import njit
@njit
def apply_ul_nb(value, lower, upper):
    entry_signal = np.full(value.shape, np.nan, dtype=np.bool_)
    exit_signal = np.full(value.shape, np.nan, dtype=np.bool_)
    for col in range(value.shape[1]):
        exit_signal[:,col] = value[:,col] <= lower
        entry_signal[:,col] = value[:,col] >= upper
            
    return entry_signal, exit_signal

def get_ULInd():
    return vbt.IndicatorFactory(
        class_name = 'UL',
        input_names = ['value'],
        param_names = ['lower', 'upper'],
        output_names = ['entry_signal', 'exit_signal']
    ).from_apply_func(apply_ul_nb)

class CSPRStrategy0(BaseStrategy):
    '''CSPR strategy'''
    _name = "CSPR"
    param_def = [
            {
            "name": "upper",
            "type": "int",
            "min":  100,
            "max":  500,
            "step": 100   
            },
            {
            "name": "lower",
            "type": "int",
            "min":  -500,
            "max":  -100,
            "step": 100   
            },
        ]


    def run(self, output_bool=False):
        ohlcv = self.stock_df[0][1]
        uppers = self.param_dict['upper']
        lowers = self.param_dict['lower']
        
        prScore = 0
        for pattern in talib.get_function_groups()['Pattern Recognition']:
            PRecognizer = vbt.IndicatorFactory.from_talib(pattern)
            pr = PRecognizer.run(ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close'])
            prScore = prScore + pr.integer
        
        #Don't look into the future
        prScore = prScore.vbt.fshift(1)

        ul_indicator = get_ULInd().run(prScore, lower=lowers, upper=uppers,\
                    param_product=True)
        entries = ul_indicator.entry_signal
        exits = ul_indicator.exit_signal

        pf = vbt.Portfolio.from_signals(
                close=ohlcv['close'], entries=entries, exits=exits, open=ohlcv['open'],
                fees=0.001, slippage=0.001,freq='1D')

        if len(uppers) > 1:
            if output_bool:
                # Draw all window combinations as a heatmap
                st.plotly_chart(
                    pf.total_return().vbt.heatmap(
                        x_level='ul_lower', 
                        y_level='ul_upper',
                    )
                )
                idxmax = (pf.total_return().idxmax())

            idxmax = (pf.total_return().idxmax())
            pf = pf[idxmax]
            self.param_dict = dict(zip(['lower', 'upper'], [int(idxmax[0]), int(idxmax[1])]))        
        self.pf = pf
   
class CSPRStrategy(BaseStrategy):
    '''CSPR strategy'''
    _name = "CSPR"
    param_def = [
            {
            "name": "pattern",
            "type": "int",
            "min":  1,
            "max":  3,
            "step": 0   
            },
        ]

    def run(self, output_bool=False):
        ohlcv = self.stock_dfs[0][1]
        patterns = self.param_dict['pattern']
        PR_list = talib.get_function_groups()['Pattern Recognition']

        prScore_df = pd.DataFrame()
        idx_list = []

        for idx, pattern in enumerate(PR_list):
            PRecognizer = vbt.IndicatorFactory.from_talib(pattern)
            pr = PRecognizer.run(ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close'])
            prScore = pr.integer
            if (prScore!=0 & pd.isnull(prScore)).sum() > 0:
            #Don't look into the future
                prScore = prScore.vbt.fshift(1)
                prScore_df[str(idx)] = prScore
                idx_list.append(idx)

        if type(patterns[0]) == str:
            prCombs = [tuple(PR_list.index(s) for s in patterns[0].split(','))]
        else:
            number = patterns[0]
            prCombs = list(combinations((idx_list), number))

        entries = pd.DataFrame()
        exits = pd.DataFrame()
        for comb in prCombs:
            prScoreCombs = prScore_df.loc[:, list(map(str, comb))].sum(axis= 1)
            entries[comb] = prScoreCombs > 0
            exits[comb] = prScoreCombs < 0

        pf = vbt.Portfolio.from_signals(
                close=ohlcv['close'], entries=entries, exits=exits, open=ohlcv['open'],
                fees=0.001, slippage=0.001,freq='1D')

        if len(pf.total_return()) > 0:
            SRs = pf.sharpe_ratio()
            idxmax = SRs[SRs != np.inf].idxmax()
            pf = pf[idxmax]
            self.param_dict = {'pattern': ','.join(PR_list[i] for i in idxmax)}
        self.pf = pf
        return True