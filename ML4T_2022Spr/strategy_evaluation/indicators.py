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


def momentum(symbol,
             sd=dt.datetime(2008, 1, 1),
             ed=dt.datetime(2009, 12, 31),
             n=20):

    df_prices = get_data([symbol], pd.date_range(sd, ed))
    df_prices = df_prices.drop(['SPY'], axis=1)

    df_momentum = (df_prices / df_prices.shift(n)) - 1

    return df_momentum


def SMA(symbol,
        sd,
        ed,
        n=50):

    df_prices = get_data([symbol], pd.date_range(sd, ed))
    df_prices = df_prices.drop(['SPY'], axis=1)
    df_prices = df_prices / df_prices.iloc[0]

    df_sma = df_prices.rolling(window=n, min_periods=1).mean()

    return df_sma


def stochastic(symbol,
               sd=dt.datetime(2008, 1, 1),
               ed=dt.datetime(2009, 12, 31),
               n=14):
    df_prices = get_data([symbol], pd.date_range(sd, ed))
    df_prices = df_prices.drop(['SPY'], axis=1)

    df_sto = 100 * (df_prices - df_prices.rolling(window=n, min_periods=1).min()) / (
                df_prices.rolling(window=n, min_periods=1).max() - df_prices.rolling(window=n, min_periods=1).min())
    df_sto.iloc[0] = 0

    return df_sto

def perc_B(symbol,
           sd,
           ed,
           n=20):

    sma = SMA(symbol, sd, ed, n=n)
    df_prices = get_data([symbol], pd.date_range(sd, ed))
    df_prices = df_prices.drop(['SPY'], axis=1)
    df_prices = df_prices / df_prices.iloc[0]

    rstd = df_prices.rolling(window=n, min_periods=1).std()
    upper = sma + 2*rstd
    lower = sma - 2*rstd

    B = (df_prices - lower) / (upper - lower)

    return B

def TSI(symbol,
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31)):

    df_prices = get_data([symbol], pd.date_range(sd, ed))
    df_prices = df_prices.drop(['SPY'], axis=1)
    # df_prices = df_prices / df_prices.iloc[0]

    PC = df_prices - df_prices.shift(1)
    PC.iloc[0] = 0
    first = PC.ewm(span=25).mean()
    second = first.ewm(span=13).mean()

    PC_a = (df_prices - df_prices.shift(1)).abs()
    PC_a.iloc[0] = 0
    first_a = PC_a.ewm(span=25).mean()
    second_a = first_a.ewm(span=13).mean()

    TSI = 100 * (second / second_a)

    return TSI


if __name__ == "__main__":
    pass
