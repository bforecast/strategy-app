import numpy as np
import pandas as pd
import talib
from itertools import combinations

import vectorbt as vbt

from .base import BaseStrategy
from utils.vbt import plot_Histogram

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

class CSPR0Strategy(BaseStrategy):
    '''CSPR strategy0
        find Candle_Stick_Pattern_Recognitions' Score to buy and sell
    '''
    _name = "CSPR0"
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

    @vbt.cached_method
    def run(self, output_bool=False, calledby='add')->bool:
        ohlcv = self.stock_df[0][1]
        close_price = self.stock_dfs[0][1].close
        open_price = self.stock_dfs[0][1].open
        uppers = self.param_dict['upper']
        lowers = self.param_dict['lower']
        
        prScore = 0
        for pattern in talib.get_function_groups()['Pattern Recognition']:
            PRecognizer = vbt.IndicatorFactory.from_talib(pattern)
            pr = PRecognizer.run(ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close'])
            prScore = prScore + pr.integer
        
        ul_indicator = get_ULInd().run(prScore, lower=lowers, upper=uppers,\
                    param_product=True)
        entries = ul_indicator.entry_signal
        exits = ul_indicator.exit_signal
        #Don't look into the future
        entries = entries.vbt.signals.fshift()
        exits = exits.vbt.signals.fshift()

        if self.param_dict['WFO']:
            entries, exits = self.maxSR_WFO(close_price, entries, exits, 'y', 1)
            pf = vbt.Portfolio.from_signals(close=close_price, open=open_price, entries=entries, exits=exits, **self.pf_kwargs)
            self.param_dict = {'WFO': True}
        else:
            pf = vbt.Portfolio.from_signals(close=close_price, open=open_price, entries=entries, exits=exits, **self.pf_kwargs)
            if calledby == 'add':
                idxmax = (pf.total_return().idxmax())
                pf = pf[idxmax]
                self.param_dict = dict(zip(['lower', 'upper'], [int(idxmax[0]), int(idxmax[1])]))        
        self.pf = pf
   
class CSPRStrategy(BaseStrategy):
    '''CSPR strategy
        find N Candle_Stick_Pattern_Recognitions' combinations
    '''
    
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

    def run(self, output_bool=False, calledby='add')->bool:
        close_price = self.stock_dfs[0][1].close
        open_price = self.stock_dfs[0][1].open
        high_price = self.stock_dfs[0][1].high
        low_price = self.stock_dfs[0][1].low
        patterns = self.param_dict['pattern']
        PR_list = talib.get_function_groups()['Pattern Recognition']

        prScore_df = pd.DataFrame()
        idx_list = []

        for idx, pattern in enumerate(PR_list):
            PRecognizer = vbt.IndicatorFactory.from_talib(pattern)
            pr = PRecognizer.run(open_price, high_price, low_price, close_price)
            prScore = pr.integer
            if (prScore!=0 & pd.isnull(prScore)).sum() > 0:
                prScore_df[str(idx)] = prScore
                idx_list.append(idx)

        if type(patterns[0]) == str:
            # update call
            prCombs = [tuple(PR_list.index(s) for s in patterns[0].split(','))]
        else:
            # maxSR call
            number = patterns[0]
            prCombs = list(combinations((idx_list), number))

        entries = pd.DataFrame()
        exits = pd.DataFrame()
        for comb in prCombs:
            prScoreCombs = prScore_df.loc[:, list(map(str, comb))].sum(axis= 1)
            entries[comb] = prScoreCombs > 0
            exits[comb] = prScoreCombs < 0
        #Don't look into the future
        entries = entries.vbt.signals.fshift()
        exits = exits.vbt.signals.fshift()

        if self.param_dict['WFO']:
            entries, exits = self.maxSR_WFO(close_price, entries, exits, 'y', 1)
            pf = vbt.Portfolio.from_signals(close=close_price, open=open_price, entries=entries, exits=exits, **self.pf_kwargs)
            self.param_dict = {'WFO': True}
        else:
            pf = vbt.Portfolio.from_signals(close=close_price, open=open_price, entries=entries, exits=exits, **self.pf_kwargs)
            if  len(pf.total_return()) > 0:
                SRs = pf.sharpe_ratio()
                idxmax = SRs[SRs != np.inf].idxmax()
                if output_bool:
                    plot_Histogram(close_price, pf, idxmax)
                pf = pf[idxmax]
                self.param_dict = {'pattern': ','.join(PR_list[i] for i in idxmax)}
        self.pf = pf
        return True


class CSPR5Strategy(BaseStrategy):
    '''CSPR5 strategy
        The 5 Most Powerful Candlestick Patterns
    '''
    _name = "CSPR5"
    desc = "**The 5 Most Powerful Candlestick Patterns ** <br>   \
            &emsp; Which candlestick pattern is most reliable? <br>    \
            &emsp; Many patterns are preferred and deemed the most reliable by different traders. Some of the most popular are: bullish/bearish engulfing lines; bullish/bearish long-legged doji; and bullish/bearish abandoned baby top and bottom. In the meantime, many neutral potential reversal signals—e.g., doji and spinning tops—will appear that should put you on the alert for the next directional move.  \
            reference: <br>\
            &emsp; https://www.investopedia.com/articles/active-trading/092315/5-most-powerful-candlestick-patterns.asp"
    param_def = []

    def run(self, output_bool=False, calledby='add')->bool:
        close_price = self.stock_dfs[0][1].close
        open_price = self.stock_dfs[0][1].open
        high_price = self.stock_dfs[0][1].high
        low_price = self.stock_dfs[0][1].low
        if 'pattern' in self.param_dict.keys():
            patterns = self.param_dict['pattern']
        else:
            patterns = []
        PR_list = ['CDLHAMMER', 'CDLMORNINGSTAR', 'CDL3WHITESOLDIERS', 
                    'CDLSHOOTINGSTAR', 'CDLEVENINGSTAR',
                    'CDL3BLACKCROWS', 'CDLENGULFING', 'CDL3OUTSIDE']

        
        prScore_df = pd.DataFrame()
        idx_list = []

        for idx, pattern in enumerate(PR_list):
            PRecognizer = vbt.IndicatorFactory.from_talib(pattern)
            pr = PRecognizer.run(open_price, high_price, low_price, close_price)
            prScore = pr.integer
            if (prScore!=0 & pd.isnull(prScore)).sum() > 0:
                prScore_df[str(idx)] = prScore
                idx_list.append(idx)
        
        if len(patterns) >0 and type(patterns[0]) == str:
            # update call
            prCombs = [tuple(PR_list.index(s) for s in patterns[0].split(','))]
        else:
            # maxSR call
            number = len(PR_list)
            prCombs = []
            for n in range(number):
                prCombs += list(combinations((idx_list), n+1))
        entries = pd.DataFrame()
        exits = pd.DataFrame()
        for comb in prCombs:
            prScoreCombs = prScore_df.loc[:, list(map(str, comb))].sum(axis= 1)
            entries[comb] = prScoreCombs > 0
            exits[comb] = prScoreCombs < 0

        #Don't look into the future
        entries = entries.vbt.signals.fshift()
        exits = exits.vbt.signals.fshift()

        if self.param_dict['WFO']:
            entries, exits = self.maxSR_WFO(close_price, entries, exits, 'y', 1)
            pf = vbt.Portfolio.from_signals(close=close_price, open=open_price, entries=entries, exits=exits, **self.pf_kwargs)
            self.param_dict = {'WFO': True}
        else:
            pf = vbt.Portfolio.from_signals(close=close_price, open=open_price, entries=entries, exits=exits, **self.pf_kwargs)
            if len(pf.total_return()) > 0:
                SRs = pf.sharpe_ratio()
                idxmax = SRs[SRs != np.inf].idxmax()
                if output_bool:
                    plot_Histogram(close_price, pf, idxmax)
                pf = pf[idxmax]
                self.param_dict = {'pattern': ','.join(PR_list[i] for i in idxmax)}
        self.pf = pf
        return True