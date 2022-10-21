import numpy as np
import pandas as pd
from datetime import timezone
from numba import njit


import streamlit as st
import vectorbt as vbt

from .base import BaseStrategy

def ecdf_nb(arr):
    result_arr = np.full_like(arr, np.nan, dtype=np.float_)
    result_arr = arr.copy()
    for i in range(arr.shape[0]):
        result_arr[i] = ((arr <= arr[i]).mean())
    return result_arr

@njit
def apply_PETOR0_nb(pettm, tor,  pe_rank, tor_rank):
    entries = np.where((pettm < pe_rank/100) & (tor < tor_rank/100), True, False)
    exits = np.where((pettm > (1-pe_rank/100)) & (tor > (1-tor_rank/100)), True, False)

    return entries, exits

@njit
def apply_PETOR_nb(pettm, tor,  pe_rankH, pe_rankL, tor_rank):
    entries = np.where((pettm < pe_rankL/100) & (tor < tor_rank/100), True, False)
    exits = np.where((pettm > pe_rankH/100) & (tor > (1-tor_rank)/100), True, False)

    return entries, exits

class PETORStrategy0(BaseStrategy):
    '''PE_TurnOverRatio strategy'''
    _name = "PETOR"
    param_def = [
            {
            "name": "pe_rank",
            "type": "int",
            "min":  2,
            "max":  25,
            "step": 2   
            },
            {
            "name": "tor_rank",
            "type": "int",
            "min":  2,
            "max":  25,
            "step": 2   
            },
        ]


    def run(self, output_bool=False):
        close_price = self.stock_df[0][1].close
        open_price = self.stock_df[0][1].open
        tor = self.stock_df[0][1].turnoverratio
        pettm = self.datas.get_pettm(self.stock_df[0][0])
        self.start_date = self.start_date.replace(tzinfo=timezone.utc)
        pettm = pettm[self.start_date: self.end_date]
        if len(pettm) == 0:
            return False
        # tor,pettm数据源不同，进行时间数据对齐
        ind_df = pd.DataFrame()
        ind_df['tor'] = tor
        ind_df['pettm'] = pettm

        tor_pers = ecdf_nb(ind_df.tor)
        pettm_pers = ecdf_nb(ind_df.pettm)

        pe_ranks = self.param_dict['pe_rank']
        tor_ranks = self.param_dict['tor_rank']

        ind = petor.run(pettm_pers, tor_pers, pe_rank=pe_ranks, tor_rank=tor_ranks,\
            param_product=True)

        # entries = (~ind.entries.isnull()).vbt.signals.fshift()
        # exits = (~ind.exits.isnull()).vbt.signals.fshift()
        entries = ind.entries
        exits = ind.exits
        pf_kwargs = dict(fees=0.001, freq='1D')
        pf = vbt.Portfolio.from_signals(
            close= close_price,
            open= open_price, 
            entries= entries, 
            exits= exits,
            **pf_kwargs
            )
        if len(pe_ranks) > 1:
            if output_bool:
                # Draw all window combinations as a 3D volume
                st.plotly_chart(
                    pf.total_return().vbt.heatmap(
                            x_level='petor_pe_rank',
                            y_level='petor_tor_rank',
                        )
                    )
                idxmax = (pf.total_return().idxmax())

            SRs = pf.sharpe_ratio()
            idxmax = SRs[SRs != np.inf].idxmax()
            pf = pf[idxmax]
            self.param_dict = dict(zip(['pe_rank', 'tor_rank'], [int(idxmax[0]), int(idxmax[1])]))        
        if output_bool:
            fig = tor_pers.vbt.plot(yaxis_title="Total Return Ratio")
            fig = pettm_pers.vbt.plot(yaxis_title="Rank Percentage", fig=fig)
            st.plotly_chart(fig)
        
        self.pf = pf
        return True
   
class PETORStrategy(BaseStrategy):
    '''PE_TurnOverRatio strategy'''
    _name = "PETOR"
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
    def run(self, output_bool=False):
        close_price = self.stock_dfs[0][1].close
        open_price = self.stock_dfs[0][1].open
        tor = self.stock_dfs[0][1].turnoverratio
        pettm = self.datas.get_pettm(self.stock_dfs[0][0])
        self.start_date = self.start_date.replace(tzinfo=timezone.utc)
        pettm = pettm[self.start_date: self.end_date]
        if len(pettm) == 0:
            return False
        # tor,pettm数据源不同，进行时间数据对齐
        ind_df = pd.DataFrame()
        ind_df['tor'] = tor
        ind_df['pettm'] = pettm

        tor_pers = ecdf_nb(ind_df.tor)
        pettm_pers = ecdf_nb(ind_df.pettm)

        pe_rankHs = self.param_dict['pe_rankH']
        pe_rankLs = self.param_dict['pe_rankL']
        tor_ranks = self.param_dict['tor_rank']

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

        entries = ind.entries.vbt.signals.fshift()
        exits = ind.exits.vbt.signals.fshift()
        pf = vbt.Portfolio.from_signals(
            close= close_price,
            open= open_price, 
            entries= entries, 
            exits= exits,
            **self.pf_kwargs
            )
        if len(pe_rankHs) > 1:
            if output_bool:
                # Draw all window combinations as a 3D volume
                st.plotly_chart(
                    pf.total_return().vbt.volume(
                            x_level='petor_pe_rankL',
                            y_level='petor_pe_rankH',
                            z_level='petor_tor_rank'
                        )
                    )
                idxmax = (pf.total_return().idxmax())

            SRs = pf.sharpe_ratio()
            if len(SRs[SRs != np.inf]) == 0:
                return False
            else:
                idxmax = SRs[SRs != np.inf].idxmax()
                pf = pf[idxmax]
                self.param_dict = dict(zip(['pe_rankH', 'pe_rankL', 'tor_rank'], [int(idxmax[0]), int(idxmax[1]), int(idxmax[2])]))        
        if output_bool:
            fig = tor_pers.vbt.plot(yaxis_title="Total Return Ratio")
            fig = pettm_pers.vbt.plot(yaxis_title="Rank Percentage", fig=fig)
            st.plotly_chart(fig)
        
        self.pf = pf
        return True