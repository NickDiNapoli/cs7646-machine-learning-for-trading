""""""
"""			     			  	 
Student Name: Nick DiNapoli  		  	   		  	  			  		 			     			  	 
GT User ID: ndinapoli6		  	   		  	  			  		 			     			  	 
GT ID: 903657316		  	   		  	  			  		 			     			  	 
"""

import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util import get_data, plot_data


def author():
    return 'ndinapoli6'


def testPolicy(symbol="JPM",
               sd=dt.datetime(2008, 1, 1),
               ed=dt.datetime(2009, 12, 31),
               sv=100000):

    df_prices = get_data([symbol], pd.date_range(sd, ed))
    df_prices = df_prices.drop(['SPY'], axis=1)

    daily_rets = df_prices.copy()
    daily_rets[1:] = (df_prices[1:] / df_prices[:-1].values) - 1
    daily_rets.iloc[0] = 0


    # print(df_prices)

    trade_dates = [daily_rets.index.values[0]]
    trade_values = [0]
    i = 1
    holdings = 0
    while (i < daily_rets.shape[0] - 1):

        if np.sign(daily_rets['JPM'][i]) == -1 and np.sign(daily_rets['JPM'][i+1]) == 1 and holdings <= 0:
            trade_dates.append(daily_rets.index.values[i])
            if holdings == 0:
                trade_values.append(1000)
                holdings += 1000
            if holdings == -1000:
                trade_values.append(2000)
                holdings += 2000

        if np.sign(daily_rets['JPM'][i]) == 1 and np.sign(daily_rets['JPM'][i+1]) == -1 and holdings >= 0:
            trade_dates.append(daily_rets.index.values[i])
            if holdings == 0:
                trade_values.append(-1000)
                holdings += -1000
            if holdings == 1000:
                trade_values.append(-2000)
                holdings += -2000

        #if np.sign(daily_rets['JPM'][i]) == np.sign(daily_rets['JPM'][i+1]):
            #pass

        i += 1

    # trade_dates.append(daily_rets.index.values[-1])
    # trade_values.append(0)

    df_trades = pd.DataFrame(trade_values, index=trade_dates)
    # print(daily_rets)
    # print(df_trades)
    # plot_data(df_prices, title="JPM stock price", xlabel="Date", ylabel="Price")

    # Benchmark
    trade_dates_benchmark = [daily_rets.index.values[0]]
    trade_values_benchmark = [1000]
    df_trades_benchmark = pd.DataFrame(trade_values_benchmark, index=trade_dates_benchmark)
    # print(df_trades_benchmark)

    return df_trades


if __name__ == "__main__":

    testPolicy()
