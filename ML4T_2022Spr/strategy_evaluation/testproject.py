""""""
"""			     			  	 
Student Name: Nick DiNapoli  		  	   		  	  			  		 			     			  	 
GT User ID: ndinapoli6		  	   		  	  			  		 			     			  	 
GT ID: 903657316		  	   		  	  			  		 			     			  	 
"""

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data, plot_data
import ManualStrategy as ms
import StrategyLearner as sl
import marketsimcode as mkt
import indicators as indc
import random

def author():
    return 'ndinapoli6'

if __name__ == "__main__":
    random.seed(10)
    ms = ms.ManualStrategy()

    #print(manual_strategy.testPolicy(symbol = "JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000))

    df_prices_all = get_data(['JPM'], pd.date_range(dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)))
    df_prices = df_prices_all[['JPM',]]
    df_prices = df_prices / df_prices.iloc[0]
    #df_prices = df_prices.drop(['SPY'], axis=1)

    """
    Part 1: Manual Strategy
    """
    # Benchmark
    trade_dates_benchmark = [dt.datetime(2008, 1, 2)]
    trade_values_benchmark = [1000]
    df_bench = pd.DataFrame(trade_values_benchmark, index=trade_dates_benchmark)
    df_bench = mkt.compute_portvals(orders_file=df_bench,
                                    sd=dt.datetime(2008, 1, 1),
                                    ed=dt.datetime(2009, 12, 31),
                                    commission=9.95,
                                    impact=0.005)
    df_bench = df_bench / df_bench.iloc[0]

    df_trades = ms.testPolicy(symbol = 'JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv = 100000)
    #df_trades.rename(columns={"Trades": "JPM"})
    df_ms = mkt.compute_portvals(orders_file=df_trades,
                                 sd=dt.datetime(2008, 1, 1),
                                 ed=dt.datetime(2009, 12, 31),
                                 commission=9.95,
                                 impact=0.005)
    #print(df_trades)
    df_ms = df_ms / df_ms.iloc[0]
    long = df_trades[df_trades['Trades'] > 0].index.tolist()
    short = df_trades[df_trades['Trades'] < 0].index.tolist()

    plt.figure(figsize=(15, 10))
    ax0 = df_bench.plot(title="Daily Portfolio Value of Manual Strategy and Benchmark (in-sample)",
                        color='purple',
                        label='Benchmark')
    df_ms.plot(color='red', label='Manual Strategy', ax=ax0)

    ax0.vlines(x=long, ymin=.7, ymax=1.4, color='blue', label='long entry points')
    ax0.vlines(x=short, ymin=.7, ymax=1.4, color='black', label='short entry points')

    ax0.set_xlabel("Date")
    ax0.set_ylabel("Value (normalized)")
    ax0.legend()


    plt.savefig('./images/Figure_0.png')
    plt.show()
    #plt.close()

    print(f"Cumulative Return of MS : {np.round_(df_ms[-1] / df_ms[0], 6) - 1}")
    print(f"Cumulative Return of Benchmark : {np.round_(df_bench[-1] / df_bench[0], 6) - 1}")
    print()

    daily_rets_jpm = df_prices.copy()
    daily_rets_jpm[1:] = (df_prices[1:] / df_prices[:-1].values) - 1
    daily_rets_jpm.iloc[0] = 0

    daily_rets_bench = df_bench.copy()
    daily_rets_bench[1:] = (df_bench[1:] / df_bench[:-1].values) - 1
    daily_rets_bench.iloc[0] = 0

    print(f"Standard Deviation of MS : {daily_rets_jpm.std()}")
    print(f"Standard Deviation of Benchmark : {daily_rets_bench.std()}")
    print()
    print(f"Average Daily Return of MS : {daily_rets_jpm.mean()}")
    print(f"Average Daily Return of Benchmark : {daily_rets_bench.mean()}")

    ####################################################################################################################

    df_prices_all_out = get_data(['JPM'], pd.date_range(dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31)))
    df_prices_out = df_prices_all_out[['JPM', ]]
    df_prices_out = df_prices_out / df_prices_out.iloc[0]
    trade_dates_benchmark_out = [dt.datetime(2010, 1, 4)]
    trade_values_benchmark_out = [1000]
    df_bench_out = pd.DataFrame(trade_values_benchmark_out, index=trade_dates_benchmark_out)
    #print(df_bench_out)
    df_bench_out = mkt.compute_portvals(orders_file=df_bench_out,
                                        sd=dt.datetime(2010, 1, 1),
                                        ed=dt.datetime(2011, 12, 31),
                                        commission=9.95,
                                        impact=0.005)
    #print(df_bench_out)
    df_bench_out = df_bench_out / df_bench_out.iloc[0]

    df_trades_out = ms.testPolicy(symbol='JPM', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)
    # df_trades.rename(columns={"Trades": "JPM"})
    df_ms_out = mkt.compute_portvals(orders_file=df_trades_out,
                                     sd=dt.datetime(2010, 1, 1),
                                     ed=dt.datetime(2011, 12, 31),
                                     commission=9.95,
                                     impact=0.005)
    #print(df_trades_out)
    df_ms_out = df_ms_out / df_ms_out.iloc[0]
    long_out = df_trades_out[df_trades_out['Trades'] > 0].index.tolist()
    short_out = df_trades_out[df_trades_out['Trades'] < 0].index.tolist()

    plt.figure(figsize=(15, 10))
    ax0 = df_bench_out.plot(title="Daily Portfolio Value of Manual Strategy and Benchmark (out-of-sample)",
                        color='purple',
                        label='Benchmark')
    df_ms_out.plot(color='red', label='Manual Strategy', ax=ax0)

    ax0.vlines(x=long_out, ymin=.8, ymax=1.1, color='blue', label='long entry points')
    ax0.vlines(x=short_out, ymin=.8, ymax=1.1, color='black', label='short entry points')

    ax0.set_xlabel("Date")
    ax0.set_ylabel("Value (normalized)")
    ax0.legend()

    plt.savefig('./images/Figure_1.png')
    plt.show()
    #plt.close()


    print(f"Cumulative Return of MS : {np.round_(df_ms_out[-1] / df_ms_out[0], 6) - 1}")
    print(f"Cumulative Return of Benchmark : {np.round_(df_bench_out[-1] / df_bench_out[0], 6) - 1}")
    print()

    daily_rets_jpm_out = df_prices_out.copy()
    daily_rets_jpm_out[1:] = (df_prices_out[1:] / df_prices_out[:-1].values) - 1
    daily_rets_jpm_out.iloc[0] = 0

    daily_rets_bench_out = df_bench_out.copy()
    daily_rets_bench_out[1:] = (df_bench_out[1:] / df_bench_out[:-1].values) - 1
    daily_rets_bench_out.iloc[0] = 0

    print(f"Standard Deviation of MS : {daily_rets_jpm_out.std()}")
    print(f"Standard Deviation of Benchmark : {daily_rets_bench_out.std()}")
    print()
    print(f"Average Daily Return of MS : {daily_rets_jpm_out.mean()}")
    print(f"Average Daily Return of Benchmark : {daily_rets_bench_out.mean()}")

    # ax3.set_xlim([0, 300])
    # ax3.set_ylim([-25, 125])



