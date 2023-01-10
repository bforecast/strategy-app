import numpy as np
import pandas as pd
from datetime import timezone
from numba import njit

import streamlit as st
import vectorbt as vbt

from .base import BaseStrategy
from utils.vbt import plot_Histogram

def ecdf_nb(arr):
    result_arr = arr.copy()
    for i in range(arr.shape[0]):
        result_arr[i] = ((arr <= arr[i]).mean())
    return result_arr

@njit
def apply_PETOR_nb(pettm, tor,  pe_rankH, pe_rankL, tor_rank):
    entries = np.where((pettm < pe_rankL/100) & (tor < tor_rank/100), True, False)
    exits = np.where((pettm > pe_rankH/100) & (tor > (1-tor_rank)/100), True, False)

    return entries, exits

class PETORStrategy(BaseStrategy):
    '''PE_TurnOverRatio strategy'''
    _name = "PETOR"
    desc = "PE_TurnOverRatio strategy"
    param_def = [
            {
            "name": "pe_rankL",
            "type": "int",
            "min":  2,
            "max":  30,
            "step": 2   
            },
            {"name": "pe_rankH",
            "type": "int",
            "min":  70,
            "max":  99,
            "step": 2 
            },
            {
            "name": "tor_rank",
            "type": "int",
            "min":  2,
            "max":  30,
            "step": 2   
            },
        ]

    @vbt.cached_method
    def run(self, calledby='add'):
        #1. initialize the variables
        if 'turnoverratio' not in self.stock_dfs[0][1].columns:
            return False
        close_price = self.stock_dfs[0][1].close
        open_price = self.stock_dfs[0][1].open
        tor = self.stock_dfs[0][1].turnoverratio
        pettm = self.datas.get_pettm(self.stock_dfs[0][0])
        self.start_date = self.start_date.replace(tzinfo=timezone.utc)
        self.end_date = self.end_date.replace(tzinfo=timezone.utc)
        pettm = pettm[self.start_date: self.end_date]
        if len(pettm) == 0:
            return False
        # tor,pettm数据源不同，进行时间数据对齐
        ind_df = pd.DataFrame()
        ind_df['tor'] = tor
        ind_df['pettm'] = pettm
        # calculate the percentage series
        tor_pers = ecdf_nb(ind_df.tor)
        pettm_pers = ecdf_nb(ind_df.pettm)
        pe_rankHs = self.param_dict['pe_rankH']
        pe_rankLs = self.param_dict['pe_rankL']
        tor_ranks = self.param_dict['tor_rank']

        #2. calculate the indicators
        petor = vbt.IndicatorFactory(
            class_name = "PETOR",
            input_names = ["pettm", "tor"],
            param_names = ["pe_rankH", "pe_rankL", "tor_rank"],
            output_names = ["entries", "exits"]
            ).from_apply_func(
                apply_PETOR_nb,
                pe_rankH = 80,
                pe_rankL = 20,
                tor_rank = 20,
                )
        ind = petor.run(pettm_pers, tor_pers, pe_rankH=pe_rankHs,  pe_rankL=pe_rankLs, 
                    tor_rank=tor_ranks, param_product=True)

        #3. remove all the name in param_def from param_dict
        for param in self.param_def:
            del self.param_dict[param['name']]

        #4. generate the vbt signal
        entries = ind.entries.vbt.signals.fshift()
        exits = ind.exits.vbt.signals.fshift()

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

                self.param_dict.update(dict(zip(['pe_rankH', 'pe_rankL', 'tor_rank'], [int(idxmax[0]), int(idxmax[1]), int(idxmax[2])])))
        
        if self.output_bool:
            fig = tor_pers.vbt.plot(yaxis_title="Total Return Ratio", yaxis_tickformat="%")
            fig = pettm_pers.vbt.plot(yaxis_title="Rank Percentage", yaxis_tickformat=".0%", fig=fig)
            st.plotly_chart(fig, use_container_width=True)
        
        self.pf = pf
        return True