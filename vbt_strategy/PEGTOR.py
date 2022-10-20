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
def apply_PEGTOR_nb(pettm, tor,  pe_rankH, pe_rankL, tor_rank):
    entries = np.where((pettm < pe_rankL/100) & (tor < tor_rank/100), True, False)
    exits = np.where((pettm > pe_rankH/100) & (tor > (1-tor_rank)/100), True, False)

    return entries, exits


class PEGTORStrategy(BaseStrategy):
    '''PE_TurnOverRatio strategy'''
    _name = "PEGTOR"
    param_def = [
            {
            "name": "peg_rankL",
            "type": "int",
            "min":  2,
            "max":  30,
            "step": 2   
            },
            {"name": "peg_rankH",
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


    def run(self, output_bool=False):
        close_price = self.stock_dfs[0][1].close
        open_price = self.stock_dfs[0][1].open
        tor = self.stock_dfs[0][1].turnoverratio
        pegttm = self.datas.get_pegttm(self.stock_dfs[0][0])
        self.start_date = self.start_date.replace(tzinfo=timezone.utc)
        pegttm = pegttm[self.start_date: self.end_date]
        if len(pegttm) == 0:
            return False
        # tor,pettm数据源不同，进行时间数据对齐
        # st.plotly_chart(pegttm.vbt.plot())
        ind_df = pd.DataFrame()
        ind_df['tor'] = tor
        ind_df['pegttm'] = pegttm

        tor_pers = ecdf_nb(ind_df.tor)
        pegttm_pers = ecdf_nb(ind_df.pegttm)

        peg_rankHs = self.param_dict['peg_rankH']
        peg_rankLs = self.param_dict['peg_rankL']
        tor_ranks = self.param_dict['tor_rank']

        pegtor = vbt.IndicatorFactory(
            class_name = "PEGTOR",
            input_names = ["pegttm", "tor"],
            param_names = ["peg_rankH", "peg_rankL", "tor_rank"],
            output_names = ["entries", "exits"]
            ).from_apply_func(
                apply_PEGTOR_nb,
                peg_rankH = 80,
                peg_rankL = 20,
                tor_rank = 20,
                )
        ind = pegtor.run(pegttm_pers, tor_pers, peg_rankH=peg_rankHs,  peg_rankL=peg_rankLs, 
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
        if len(peg_rankHs) > 1:
            if output_bool:
                # Draw all window combinations as a 3D volume
                st.plotly_chart(
                    pf.total_return().vbt.volume(
                            x_level='pegtor_peg_rankL',
                            y_level='pegtor_peg_rankH',
                            z_level='pegtor_tor_rank'
                        )
                    )
                idxmax = (pf.total_return().idxmax())

            SRs = pf.sharpe_ratio()
            if len(SRs[SRs != np.inf]) == 0:
                return False
            else:
                idxmax = SRs[SRs != np.inf].idxmax()
                pf = pf[idxmax]
                self.param_dict = dict(zip(['peg_rankH', 'peg_rankL', 'tor_rank'], [int(idxmax[0]), int(idxmax[1]), int(idxmax[2])]))        
        
        if output_bool:
            fig = tor_pers.vbt.plot(yaxis_title="Total Return Ratio")
            fig = pegttm_pers.vbt.plot(yaxis_title="Rank Percentage", fig=fig)
            st.plotly_chart(fig)
        
        self.pf = pf
        return True