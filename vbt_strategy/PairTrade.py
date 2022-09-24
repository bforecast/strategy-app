import numpy as np
import pandas as pd
import datetime
import collections
import math
import pytz
from numba import njit
import scipy.stats as stats
import vectorbt as vbt
from statsmodels.tsa.stattools import coint

import streamlit as st

from utils.vbt_nb import plot_pf
from .base import BaseStrategy


@njit
def rolling_logret_zscore_nb(a, b, window):
    """Calculate the log return spread."""
    spread = np.full_like(a, np.nan, dtype=np.float_)
    spread[1:] = np.log(a[1:] / a[:-1]) - np.log(b[1:] / b[:-1])
    zscore = np.full_like(a, np.nan, dtype=np.float_)
    for i in range(a.shape[0]):
        from_i = max(0, i + 1 - window)
        to_i = i + 1
        if i < window - 1:
            continue
        spread_mean = np.mean(spread[from_i:to_i])
        spread_std = np.std(spread[from_i:to_i])
        zscore[i] = (spread[i] - spread_mean) / spread_std
    return spread, zscore

@njit
def ols_spread_nb(a, b):
    """Calculate the OLS spread."""
    a = np.log(a)
    b = np.log(b)
    _b = np.vstack((b, np.ones(len(b)))).T
    slope, intercept = np.dot(np.linalg.inv(np.dot(_b.T, _b)), np.dot(_b.T, a))
    spread = a - (slope * b + intercept)
    return spread[-1]
    
@njit
def rolling_ols_zscore_nb(a, b, window):
    """Calculate the z-score of the rolling OLS spread."""
    spread = np.full_like(a, np.nan, dtype=np.float_)
    zscore = np.full_like(a, np.nan, dtype=np.float_)
    for i in range(a.shape[0]):
        from_i = max(0, i + 1 - window)
        to_i = i + 1
        if i < window - 1:
            continue
        spread[i] = ols_spread_nb(a[from_i:to_i], b[from_i:to_i])
        spread_mean = np.mean(spread[from_i:to_i])
        spread_std = np.std(spread[from_i:to_i])
        zscore[i] = (spread[i] - spread_mean) / spread_std
    return spread, zscore

from vectorbt.portfolio import nb as portfolio_nb
from vectorbt.base.reshape_fns import flex_select_auto_nb
from vectorbt.portfolio.enums import SizeType, Direction
from collections import namedtuple

Memory = namedtuple("Memory", ('spread', 'zscore', 'status'))
Params = namedtuple("Params", ('window', 'upper', 'lower', 'order_pct1', 'order_pct2'))

@njit
def pre_group_func_nb(c, _window, _upper, _lower, _order_pct1, _order_pct2):
    """Prepare the current group (= pair of columns)."""
    assert c.group_len == 2
    
    # In contrast to bt, we don't have a class instance that we could use to store arrays,
    # so let's create a namedtuple acting as a container for our arrays
    # ( you could also pass each array as a standalone object, but a single object is more convenient)
    spread = np.full(c.target_shape[0], np.nan, dtype=np.float_)
    zscore = np.full(c.target_shape[0], np.nan, dtype=np.float_)
    
    # Note that namedtuples aren't mutable, you can't simply assign a value,
    # thus make status variable an array of one element for an easy assignment
    status = np.full(1, 0, dtype=np.int_)
    memory = Memory(spread, zscore, status)
    
    # Treat each param as an array with value per group, and select the combination of params for this group
    window = flex_select_auto_nb(np.asarray(_window), 0, c.group, True)
    upper = flex_select_auto_nb(np.asarray(_upper), 0, c.group, True)
    lower = flex_select_auto_nb(np.asarray(_lower), 0, c.group, True)
    order_pct1 = flex_select_auto_nb(np.asarray(_order_pct1), 0, c.group, True)
    order_pct2 = flex_select_auto_nb(np.asarray(_order_pct2), 0, c.group, True)
    
    # Put all params into a container (again, this is optional)
    params = Params(window, upper, lower, order_pct1, order_pct2)
    
    # Create an array that will store our two target percentages used by order_func_nb
    # we do it here instead of in pre_segment_func_nb to initialize the array once, instead of in each row
    size = np.empty(c.group_len, dtype=np.float_)
    
    # The returned tuple is passed as arguments to the function below
    return (memory, params, size)
    

@njit
def pre_segment_func_nb(c, memory, params, size, mode):
    """Prepare the current segment (= row within group)."""
    
    # We want to perform calculations once we reach full window size
    if c.i < params.window - 1:
        size[0] = np.nan  # size of nan means no order
        size[1] = np.nan
        return (size,)
    
    # z-core is calculated using a window (=window) of spread values
    # This window can be specified as a slice
    window_slice = slice(max(0, c.i + 1 - params.window), c.i + 1)
    
    # Here comes the same as in rolling_ols_zscore_nb
    if mode == 'OLS':
        a = c.close[window_slice, c.from_col]
        b = c.close[window_slice, c.from_col + 1]
        memory.spread[c.i] = ols_spread_nb(a, b)
    elif mode == 'log_return':
        logret_a = np.log(c.close[c.i, c.from_col] / c.close[c.i - 1, c.from_col])
        logret_b = np.log(c.close[c.i, c.from_col + 1] / c.close[c.i - 1, c.from_col + 1])
        memory.spread[c.i] = logret_a - logret_b
    else:
        raise ValueError("Unknown mode")
    spread_mean = np.mean(memory.spread[window_slice])
    spread_std = np.std(memory.spread[window_slice])
    memory.zscore[c.i] = (memory.spread[c.i] - spread_mean) / spread_std
    
    # Check if any bound is crossed
    # Since zscore is calculated using close, use zscore of the previous step
    # This way we are executing signals defined at the previous bar
    # Same logic as in PairTradingStrategy
    if memory.zscore[c.i - 1] > params.upper and memory.status[0] != 1:
        size[0] = -params.order_pct1
        size[1] = params.order_pct2
        
        # Here we specify the order of execution
        # call_seq_now defines order for the current group (2 elements)
        c.call_seq_now[0] = 0
        c.call_seq_now[1] = 1
        memory.status[0] = 1
    elif memory.zscore[c.i - 1] < params.lower and memory.status[0] != 2:
        size[0] = params.order_pct1
        size[1] = -params.order_pct2
        c.call_seq_now[0] = 1  # execute the second order first to release funds early
        c.call_seq_now[1] = 0
        memory.status[0] = 2
    else:
        size[0] = np.nan
        size[1] = np.nan
        
    # Group value is converted to shares using previous close, just like in bt
    # Note that last_val_price contains valuation price of all columns, not just the current pair
    c.last_val_price[c.from_col] = c.close[c.i - 1, c.from_col]
    c.last_val_price[c.from_col + 1] = c.close[c.i - 1, c.from_col + 1]
        
    return (size,)

@njit
def order_func_nb(c, size, price, commperc):
    """Place an order (= element within group and row)."""
    
    # Get column index within group (if group starts at column 58 and current column is 59, 
    # the column within group is 1, which can be used to get size)
    group_col = c.col - c.from_col
    # if c.position_now == 0 and size[group_col] < 0:
    #     return portfolio_nb.order_nb(
    #     size=0, 
    #     price=price[c.i, c.col],
    #     size_type=SizeType.TargetPercent,
    #     fees=commperc
    # )
    # else:
    return portfolio_nb.order_nb(
            size=size[group_col], 
            price=price[c.i, c.col],
            size_type=SizeType.TargetPercent,
            fees=commperc,
            direction=Direction.LongOnly
        )

def PairTrade(ohlcv1, ohlcv2, symbol1, symbol2):
    #initialize Parameters
    window = 100
    CASH = 100000
    COMMPERC = 0.005  # 0.5%
    ORDER_PCT1 = 0.95   #0.1
    ORDER_PCT2 = 0.95   #0.1
    UPPER = stats.norm.ppf(1 - 0.05 / 2)
    LOWER = -stats.norm.ppf(1 - 0.05 / 2)
    MODE = 'OLS'  # OLS, log_return

    score, pvalue, _ = coint(ohlcv1['close'],ohlcv2['close'])
    st.write(f"P-Value is {pvalue}")

    symbol_cols = pd.Index([symbol1, symbol2], name='symbol')
    vbt_close_price = pd.concat((ohlcv1['close'], ohlcv2['close']), axis=1, keys=symbol_cols)
    vbt_open_price = pd.concat((ohlcv1['open'], ohlcv2['open']), axis=1, keys=symbol_cols)

    windows = np.arange(10, 105, 5)
    uppers = np.arange(1.5, 2.2, 0.1)
    lowers = -1 * np.arange(1.5, 2.2, 0.1)

    def simulate_mult_from_order_func(windows, uppers, lowers):
        """Simulate multiple parameter combinations using `Portfolio.from_order_func`."""
        # Build param grid
        param_product = vbt.utils.params.create_param_product([windows, uppers, lowers])
        param_tuples = list(zip(*param_product))
        param_columns = pd.MultiIndex.from_tuples(param_tuples, names=['window', 'upper', 'lower'])
        
        # We need two price columns per param combination
        vbt_close_price_mult = vbt_close_price.vbt.tile(len(param_columns), keys=param_columns)
        vbt_open_price_mult = vbt_open_price.vbt.tile(len(param_columns), keys=param_columns)
        
        return vbt.Portfolio.from_order_func(
            vbt_close_price_mult,
            order_func_nb, 
            vbt_open_price_mult.values, COMMPERC,  # *args for order_func_nb
            pre_group_func_nb=pre_group_func_nb, 
            pre_group_args=(
                np.array(param_product[0]), 
                np.array(param_product[1]), 
                np.array(param_product[2]), 
                ORDER_PCT1, 
                ORDER_PCT2
            ),
            pre_segment_func_nb=pre_segment_func_nb, 
            pre_segment_args=(MODE,),
            fill_pos_record=False,
            init_cash=CASH,
            cash_sharing=True, 
            group_by=param_columns.names,
            freq='d'
        )

    vbt_pf_mult = simulate_mult_from_order_func(windows, uppers, lowers)
    # Draw all window combinations as a 3D volume
    st.plotly_chart(
        vbt_pf_mult.total_return().vbt.volume(
                x_level='upper',
                y_level='lower',
                z_level='window',

                trace_kwargs=dict(
                    colorbar=dict(
                        title='Total return', 
                        tickformat='%'
                    )
                )
            )
        )
    # Max Sharpe_ratio Parameter    
    idxmax = (vbt_pf_mult.sharpe_ratio().idxmax())
    return_pf = vbt_pf_mult[idxmax]
    plot_pf(return_pf)

    param_dict = dict(zip(['window', 'upper', 'lower'], [int(idxmax[0]), round(idxmax[1], 2), round(idxmax[2]), 2]))
    st.write(param_dict)
    return param_dict, return_pf

    
class PairTradeStrategy(BaseStrategy):
    '''PairTrade strategy'''
    _name = "PairTrade"
    param_dict = {}

    def run(self, output_bool=False):
        #initialize Parameters
        window = 100
        CASH = 100000
        COMMPERC = 0.005  # 0.5%
        ORDER_PCT1 = 0.95   #0.1
        ORDER_PCT2 = 0.95   #0.1
        MODE = 'OLS'  # OLS, log_return

        windows = self.param_dict['window']
        uppers = self.param_dict['upper']
        lowers = self.param_dict['lower']

        ohlcv1 = self.ohlcv_list[0][1]
        ohlcv2 = self.ohlcv_list[1][1]
        symbol1 = self.ohlcv_list[0][0]
        symbol2 = self.ohlcv_list[1][0]
        
        symbol_cols = pd.Index([symbol1, symbol2], name='symbol')
        vbt_close_price = pd.concat((ohlcv1['close'], ohlcv2['close']), axis=1, keys=symbol_cols)
        vbt_open_price = pd.concat((ohlcv1['open'], ohlcv2['open']), axis=1, keys=symbol_cols)


        def simulate_mult_from_order_func(windows, uppers, lowers):
            """Simulate multiple parameter combinations using `Portfolio.from_order_func`."""
            # Build param grid
            param_product = vbt.utils.params.create_param_product([windows, uppers, lowers])
            param_tuples = list(zip(*param_product))
            param_columns = pd.MultiIndex.from_tuples(param_tuples, names=['window', 'upper', 'lower'])
            
            # We need two price columns per param combination
            vbt_close_price_mult = vbt_close_price.vbt.tile(len(param_columns), keys=param_columns)
            vbt_open_price_mult = vbt_open_price.vbt.tile(len(param_columns), keys=param_columns)
            
            return vbt.Portfolio.from_order_func(
                vbt_close_price_mult,
                order_func_nb, 
                vbt_open_price_mult.values, COMMPERC,  # *args for order_func_nb
                pre_group_func_nb=pre_group_func_nb, 
                pre_group_args=(
                    np.array(param_product[0]), 
                    np.array(param_product[1]), 
                    np.array(param_product[2]), 
                    ORDER_PCT1, 
                    ORDER_PCT2
                ),
                pre_segment_func_nb=pre_segment_func_nb, 
                pre_segment_args=(MODE,),
                fill_pos_record=False,
                init_cash=CASH,
                cash_sharing=True, 
                group_by=param_columns.names,
                freq='d'
            )

        vbt_pf_mult = simulate_mult_from_order_func(windows, uppers, lowers)
        if output_bool:
            # Draw all window combinations as a 3D volume
            st.plotly_chart(
                vbt_pf_mult.total_return().vbt.volume(
                        x_level='upper',
                        y_level='lower',
                        z_level='window',

                        trace_kwargs=dict(
                            colorbar=dict(
                                title='Total return', 
                                tickformat='%'
                            )
                        )
                    )
                )
        if len(windows) > 1:
            # Max Sharpe_ratio Parameter    
            idxmax = (vbt_pf_mult.sharpe_ratio().idxmax())
            pf = vbt_pf_mult[idxmax]
            self.param_dict = dict(zip(['window', 'upper', 'lower'], [int(idxmax[0]), round(idxmax[1], 4), round(idxmax[2], 4)]))
        else:
            pf =vbt_pf_mult
        return pf

    def maxSR(self, output_bool=False):
        self.param_dict = {
            "window":   np.arange(10, 105, 5),
            'upper':    np.arange(1.5, 2.2, 0.1),
            'lower':    np.arange(1.5, 2.2, 0.1)
        }

        # score, pvalue, _ = coint(ohlcv1['close'], ohlcv2['close'])
        # if output_bool:
        #     st.write(f"P-Value is {pvalue}")

        pf = self.run(output_bool)
        if output_bool:
            plot_pf(pf)
       
        return self.param_dict, pf
    
