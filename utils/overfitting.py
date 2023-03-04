'''
reference: https://github.com/finlab-python/finlab_crypto
'''
from statsmodels.distributions.empirical_distribution import ECDF
import itertools as itr
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import math

import vectorbt as vbt

sharpe_ratio = lambda r: r.mean() / (r.std()+0.0000001) * (252 ** 0.5)
RARM_dict = {
    "sharpe_ratio": lambda r: r.mean() / (r.std()+0.0000001) * (252 ** 0.5),
    "sortino_ratio": lambda r: r.mean() / (r[r<0].std()+0.0000001) * (252 ** 0.5),
    "calmar_ratio": lambda r: r.mean()*255/abs(),
}


class CSCV(object):
    """Combinatorially symmetric cross-validation algorithm.

    Calculate backtesting about overfitting probability distribution and performance degradation.
        基于组合对称交叉验证(CSCV)框架计算策略的过拟合概率(PBO)的方法，这是一种较好的检验过拟合的方法。其核心思想并非检验某个参数是否过拟合，而是某类策略的参 数选取方法是否容易带来过拟合。具体到本策略中，CSCV框架测算的是针对直接牛熊策 略的网格寻优方法是否容易带来过拟合。在过拟合检验中，过拟合概率 PBO 的定义为： 样本内最优参数在样本外的夏普比率排名位于后 50%的概率。一般情况下 PBO 概率小于
        50%可以认为过拟合概率不高。PBO 越小，回测过拟合概率越低。
    Attributes:
        n_bins:A int of CSCV algorithm bin size to control overfitting calculation.Default is 10.
        RARM:A "Risk Adjusted Return Method" vbt accessors 'method of in sample(is) and out of sample(oos) return benchmark algorithm.Default is sharpe_ratio.

    """
    def __init__(self, n_bins=10, RARM="sharpe_ratio"):
        self.n_bins = n_bins
        if RARM=="annualized_return":
            self.RARM = 'annualized'
        elif  RARM=="information_ratio":
            print('vbt_accessors_information_ratio error called. replaced by sharpe_ratio')
            self.RARM = 'sharpe_ratio'
        else:
            self.RARM = RARM

        self.bins_enumeration = [set(x) for x in itr.combinations(np.arange(10), 10 // 2)]
        self.Rs = [pd.Series(dtype=float) for i in range(len(self.bins_enumeration))]
        self.R_bars = [pd.Series(dtype=float) for i in range(len(self.bins_enumeration))]

    def add_daily_returns(self, daily_returns):
        """Add daily_returns in algorithm.

        Args:
          daily_returns: A dataframe of trading daily_returns.

        """
        bin_size = daily_returns.shape[0] // self.n_bins
        bins = [daily_returns.iloc[i*bin_size: (i+1) * bin_size] for i in range(self.n_bins)]

        for set_id, is_set in enumerate(self.bins_enumeration):
            oos_set = set(range(10)) - is_set
            is_returns = pd.concat([bins[i] for i in is_set])
            oos_returns = pd.concat([bins[i] for i in oos_set])
            R = eval(f"is_returns.vbt.returns(freq='d').{self.RARM}()")
            R_bar = eval(f"oos_returns.vbt.returns(freq='d').{self.RARM}()")
            self.Rs[set_id] = self.Rs[set_id].append(R)
            self.R_bars[set_id] = self.R_bars[set_id].append(R_bar)

    def estimate_overfitting(self, plot=False):
        """Estimate overfitting probability.

        Generate the result on Combinatorially symmetric cross-validation algorithm.
        Display related analysis charts.

        Args:
          plot: A bool of control plot display. Default is False.

        Returns:
          A dict of result include:
          pbo_test: A float of overfitting probability.
          logits: A float of estimated logits of OOS rankings.
          R_n_star: A list of IS performance of th trategies that has the best ranking in IS.
          R_bar_n_star: A list of find the OOS performance of the strategies that has the best ranking in IS.
          dom_df: A dataframe of optimized_IS, non_optimized_OOS data.

        """
        # calculate strategy performance in IS(R_df) and OOS(R_bar_df)
        R_df = pd.DataFrame(self.Rs)
        R_bar_df = pd.DataFrame(self.R_bars)

        # calculate ranking of the strategies
        R_rank_df = R_df.rank(axis=1, ascending=False, method='first')
        R_bar_rank_df = R_bar_df.rank(axis=1, ascending=False, method='first')

        # find the IS performance of th trategies that has the best ranking in IS
        r_star_series = (R_df * (R_rank_df == 1)).unstack().dropna()
        r_star_series = r_star_series[r_star_series != 0].sort_index(level=-1)

        # find the OOS performance of the strategies that has the best ranking in IS
        r_bar_star_series = (R_bar_df * (R_rank_df == 1)).unstack().dropna()
        r_bar_star_series = r_bar_star_series[r_bar_star_series != 0].sort_index(level=-1)

        # find the ranking of strategies which has the best ranking in IS
        r_bar_rank_series = (R_bar_rank_df * (R_rank_df == 1)).unstack().dropna()
        r_bar_rank_series = r_bar_rank_series[r_bar_rank_series != 0].sort_index(level=-1)

        # probability of overfitting

        # estimate logits of OOS rankings
        logits = (1-((r_bar_rank_series)/(len(R_df.columns)+1))).map(lambda p: math.log(p/(1-p)))
        prob = (logits < 0).sum() / len(logits)

        # stochastic dominance

        # caluclate
        if len(r_bar_star_series) != 0:
            y = np.linspace(
                min(r_bar_star_series), max(r_bar_star_series), endpoint=True, num=1000
            )

            # build CDF performance of best candidate in IS
            R_bar_n_star_cdf = ECDF(r_bar_star_series.values)
            optimized = R_bar_n_star_cdf(y)

            # build CDF performance of average candidate in IS
            R_bar_mean_cdf = ECDF(R_bar_df.median(axis=1).values)
            non_optimized = R_bar_mean_cdf(y)

            #
            dom_df = pd.DataFrame(
                dict(optimized_IS=optimized, non_optimized_OOS=non_optimized)
            , index=y)
            dom_df["SD2"] = -(dom_df.non_optimized_OOS - dom_df.optimized_IS).cumsum()
        else:
            dom_df = pd.DataFrame(columns=['optimized_IS', 'non_optimized_OOS', 'SD2'])

        ret = {
            'pbo_test': (logits < 0).sum() / len(logits),
            'logits': logits.to_list(),
            'R_n_star': r_star_series.to_list(),
            'R_bar_n_star': r_bar_star_series.to_list(),
            'dom_df': dom_df,
        }
        return ret

    @staticmethod
    def plot(results):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             sharey=False, sharex=False, constrained_layout=False)
        fig.subplots_adjust(bottom=0.5)
        fig.suptitle('Combinatorially Symmetric Cross-validation')

        pbo_test = round(results['pbo_test'] * 100, 2)
        axes[0].title.set_text(f'Probability of overfitting: {pbo_test} %')
        axes[0].hist(x=[l for l in results['logits'] if l > -10000], bins='auto')
        axes[0].set_xlabel('Logits')
        axes[0].set_ylabel('Frequency')

        # performance degradation
        axes[1].title.set_text('Performance degradation')
        x, y = pd.DataFrame([results['R_n_star'], results['R_bar_n_star']]).dropna(axis=1).values
        sns.regplot(x, y, ax=axes[1])
        axes[1].set_xlabel('In-sample Performance')
        axes[1].set_ylabel('Out-of-sample Performance')

        # first and second Stochastic dominance
        axes[2].title.set_text('Stochastic dominance')
        if len(results['dom_df']) != 0: results['dom_df'].plot(ax=axes[2], secondary_y=['SD2'])
        axes[2].set_xlabel('Performance optimized vs non-optimized')
        axes[2].set_ylabel('Frequency')
        return fig
