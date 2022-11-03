import pandas as pd
import vectorbt as vbt
import numpy as np

def cal_vbtpf(prices, weights):
    size = pd.DataFrame.vbt.empty_like(prices, fill_value=np.nan)
    size.iloc[0] =  weights  # starting weights
    pf_kwargs = dict(fees=0.001, slippage=0.001, freq='1D')

    pf = vbt.Portfolio.from_orders(
            prices, 
            size, 
            size_type='targetpercent', 
            group_by=True,  # group of two columns
            cash_sharing=True,  # share capital between columns
            **pf_kwargs,
        )
    return pf