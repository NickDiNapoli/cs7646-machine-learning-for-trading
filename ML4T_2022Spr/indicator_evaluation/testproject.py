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
import TheoreticallyOptimalStrategy as tos
import marketsimcode as mkt
import indicators as indc

def author():
    return 'ndinapoli6'

if __name__ == "__main__":

    df_prices = get_data(['JPM'], pd.date_range(dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)))
    df_prices = df_prices.drop(['SPY'], axis=1)
    df_prices = df_prices / df_prices.iloc[0]

    """
    Part 1: Theoretical Optimal Strategy
    """
    # Benchmark
    trade_dates_benchmark = [dt.datetime(2008, 1, 2)]
    trade_values_benchmark = [1000]
    df_bench = pd.DataFrame(trade_values_benchmark, index=trade_dates_benchmark)
    df_bench = mkt.compute_portvals(orders_file=df_bench)
    df_bench = df_bench / df_bench.iloc[0]

    #df_trades = tos.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    df_jpm = mkt.compute_portvals(orders_file=tos.testPolicy())
    df_jpm = df_jpm / df_jpm.iloc[0]

    df_bench.plot(title="Daily Portfolio Value of TOS and Benchmark",
                  color='purple',
                  fontsize=12,
                  legend=True)

    df_jpm.plot(color='red', legend=True)
    plt.legend(['Benchmark', 'TOS'])

    plt.xlabel("Date")
    plt.ylabel("Value (normalized)")
    # plt.show()
    plt.savefig('./Figure_1.png')
    plt.close()

    print(f"Cumulative Return of TOS : {np.round_(df_jpm[-1]/df_jpm[0], 6) - 1}")
    print(f"Cumulative Return of Benchmark : {np.round_(df_bench[-1]/df_bench[0], 6) - 1}")
    print()

    daily_rets_jpm = df_jpm.copy()
    daily_rets_jpm[1:] = (df_jpm[1:] / df_jpm[:-1].values) - 1
    daily_rets_jpm.iloc[0] = 0

    daily_rets_bench = df_bench.copy()
    daily_rets_bench[1:] = (df_bench[1:] / df_bench[:-1].values) - 1
    daily_rets_bench.iloc[0] = 0

    print(f"Standard Deviation of TOS : {daily_rets_jpm.std()}")
    print(f"Standard Deviation of Benchmark : {daily_rets_bench.std()}")
    print()
    print(f"Average Daily Return of TOS : {daily_rets_jpm.mean()}")
    print(f"Average Daily Return of Benchmark : {daily_rets_bench.mean()}")

    """
    Part 2: Technical Indicators
    """

    df_momentum = indc.momentum()
    ax1 = df_momentum.plot(title="Momentum of JPM",
                           color='blue',
                           label='Momentum')

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Value")
    ax1.legend(['Momentum'])
    # plt.show()
    plt.savefig('./Figure_2.png')
    plt.close()

    df_sma = indc.SMA()
    df_price_sma_ratio = df_prices / df_sma
    df_price_sma_ratio.columns = ['JPM']
    plt.figure(figsize=(8, 6))
    ax2 = df_sma['JPM'].plot(title="SMA of JPM",
                           color='red',
                           label='SMA')

    df_prices['JPM'].plot(color='purple', label='JPM Price', ax=ax2)
    df_price_sma_ratio['JPM'].plot(color='blue', label='Price/SMA Ratio', ax=ax2)

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Value (Normalized)")
    ax2.legend()
    # plt.show()
    plt.savefig('./Figure_3.png')
    plt.close()

    df_sto = indc.stochastic()
    df_sto_avg3 = df_sto.rolling(window=3, min_periods=1).mean()
    plt.figure(figsize=(10, 5))
    ax3 = df_sto['JPM'].plot(title="Stochastic Indicator/Oscillator of JPM",
                             color='blue',
                             label='Stochastic Indicator')

    df_sto_avg3['JPM'].plot(color='red', label='3-day MA', ax=ax3)
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Value (%)")
    # ax3.set_xlim([0, 300])
    ax3.set_ylim([-25, 125])
    ax3.legend()
    # plt.show()
    plt.savefig('./Figure_4.png')
    plt.close()

    df_B = indc.perc_B()
    plt.figure(figsize=(10, 5))
    ax4 = df_B['JPM'].plot(title="%B Indicator of JPM",
                             color='blue',
                             label='%B Indicator')

    ax4.set_xlabel("Date")
    ax4.set_ylabel("Value")
    # ax3.set_xlim([0, 300])
    # ax3.set_ylim([-25, 125])
    ax4.legend()
    # plt.show()
    plt.savefig('./Figure_5.png')
    plt.close()

    df_TSI = indc.TSI()
    # plt.figure(figsize=(10, 5))
    ax5 = df_TSI['JPM'].plot(title="True Strength Index of JPM",
                           color='blue',
                           label='TSI')

    ax5.set_xlabel("Date")
    ax5.set_ylabel("Value")
    # ax3.set_xlim([0, 300])
    # ax3.set_ylim([-25, 125])
    ax5.legend()
    # plt.show()
    plt.savefig('./Figure_6.png')
    plt.close()

