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

def author():
    return 'ndinapoli6'

if __name__ == "__main__":

    df_prices_all = get_data(['JPM'], pd.date_range(dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)))
    df_prices = df_prices_all[['JPM', ]]
    df_prices = df_prices / df_prices.iloc[0]

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

    # Manual Strategy
    ms = ms.ManualStrategy()

    df_trades = ms.testPolicy(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)

    df_ms = mkt.compute_portvals(orders_file=df_trades,
                                 sd=dt.datetime(2008, 1, 1),
                                 ed=dt.datetime(2009, 12, 31),
                                 commission=9.95,
                                 impact=0.005)

    df_ms = df_ms / df_ms.iloc[0]

    # Strategy Learner

    learner = sl.StrategyLearner(verbose=False, impact=0.005, commission=9.95)  # constructor
    learner.add_evidence(symbol="JPM",
                               sd=dt.datetime(2008, 1, 1),
                               ed=dt.datetime(2009, 12, 31),
                               sv=100000)  # training phase

    df_trades_sl = learner.testPolicy(symbol="JPM",
                                   sd=dt.datetime(2008, 1, 1),
                                   ed=dt.datetime(2009, 12, 31),
                                   sv=100000)  # testing phase

    df_sl = mkt.compute_portvals(orders_file=df_trades_sl,
                                 sd=dt.datetime(2008, 1, 1),
                                 ed=dt.datetime(2009, 12, 31),
                                 commission=9.95,
                                 impact=0.005)

    df_sl = df_sl / df_sl.iloc[0]

    plt.figure(figsize=(15, 10))
    ax0 = df_bench.plot(title="Daily Portfolio Value of Strategy Learner, Manual Strategy, and Benchmark (in-sample)",
                        color='purple',
                        label='Benchmark')

    df_ms.plot(color='red', label='Manual Strategy', ax=ax0)
    df_sl.plot(color='blue', label='Strategy Learner', ax=ax0)

    ax0.set_xlabel("Date")
    ax0.set_ylabel("Value (normalized)")
    ax0.legend()

    plt.savefig('./images/Figure_2.png')
    plt.show()
    # plt.close()

    ####################################################################################################################

    df_prices_all = get_data(['JPM'], pd.date_range(dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31)))
    df_prices = df_prices_all[['JPM', ]]
    df_prices = df_prices / df_prices.iloc[0]

    # Benchmark
    trade_dates_benchmark = [dt.datetime(2010, 1, 4)]
    trade_values_benchmark = [1000]
    df_bench = pd.DataFrame(trade_values_benchmark, index=trade_dates_benchmark)
    df_bench = mkt.compute_portvals(orders_file=df_bench,
                                    sd=dt.datetime(2010, 1, 1),
                                    ed=dt.datetime(2011, 12, 31),
                                    commission=9.95,
                                    impact=0.005)
    df_bench = df_bench / df_bench.iloc[0]

    # Manual Strategy

    df_trades = ms.testPolicy(symbol='JPM', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)

    df_ms = mkt.compute_portvals(orders_file=df_trades,
                                 sd=dt.datetime(2010, 1, 1),
                                 ed=dt.datetime(2011, 12, 31),
                                 commission=9.95,
                                 impact=0.005)

    df_ms = df_ms / df_ms.iloc[0]

    # Strategy Learner

    learner = sl.StrategyLearner(verbose=False, impact=0.005, commission=9.95)  # constructor
    learner.add_evidence(symbol="JPM",
                         sd=dt.datetime(2010, 1, 1),
                         ed=dt.datetime(2011, 12, 31),
                         sv=100000)  # training phase

    df_trades_sl = learner.testPolicy(symbol="JPM",
                                      sd=dt.datetime(2010, 1, 1),
                                      ed=dt.datetime(2011, 12, 31),
                                      sv=100000)  # testing phase

    df_sl = mkt.compute_portvals(orders_file=df_trades_sl,
                                 sd=dt.datetime(2010, 1, 1),
                                 ed=dt.datetime(2011, 12, 31),
                                 commission=9.95,
                                 impact=0.005)

    df_sl = df_sl / df_sl.iloc[0]

    plt.figure(figsize=(15, 10))
    ax0 = df_bench.plot(title="Daily Portfolio Value of Strategy Learner, Manual Strategy, and Benchmark (out-of-sample)",
                        color='purple',
                        label='Benchmark')

    df_ms.plot(color='red', label='Manual Strategy', ax=ax0)
    df_sl.plot(color='blue', label='Strategy Learner', ax=ax0)

    ax0.set_xlabel("Date")
    ax0.set_ylabel("Value (normalized)")
    ax0.legend()

    plt.savefig('./images/Figure_3.png')
    plt.show()
    # plt.close()