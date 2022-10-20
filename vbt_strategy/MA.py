import numpy as np

import streamlit as st
import vectorbt as vbt

from .base import BaseStrategy

class MAStrategy(BaseStrategy):
    '''MA strategy'''
    _name = "MA"
    param_def = [
            {
            "name": "window",
            "type": "int",
            "min":  2,
            "max":  101,
            "step": 1   
            },
        ]

    def run(self, output_bool=False)->bool:
        price = self.stock_dfs[0][1].close
        if "window" in self.param_dict.keys():
            window = self.param_dict['window']
            fast_ma, slow_ma = vbt.MA.run_combs(price, window=window, r=2, short_names=['fast', 'slow'])
        else:
            fast_windows = self.param_dict['fast_window']
            slow_windows = self.param_dict['slow_window']
            fast_ma = vbt.MA.run(price, fast_windows)
            slow_ma = vbt.MA.run(price, slow_windows)

        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        #Don't look into the future
        entries = entries.vbt.signals.fshift()
        exits = exits.vbt.signals.fshift()

        pf = vbt.Portfolio.from_signals(price, entries, exits, **self.pf_kwargs)

        if output_bool:
            fig = pf.total_return().vbt.heatmap(
                x_level='fast_window', y_level='slow_window', symmetric=True,
                trace_kwargs=dict(colorbar=dict(title='Total return', tickformat='%')))
            st.plotly_chart(fig)

        if "window" in self.param_dict.keys():
            SRs = pf.sharpe_ratio()
            idxmax = SRs[SRs != np.inf].idxmax()
            # st.write(idxmax)
            pf = pf[idxmax]
            self.param_dict = dict(zip(['fast_window', 'slow_window'], [int(idxmax[0]), int(idxmax[1])]))
        
        self.pf = pf
        return True
