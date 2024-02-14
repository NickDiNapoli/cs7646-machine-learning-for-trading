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
    # Strategy Learner
    for impact in np.linspace(0, 1, 5):
        learner = sl.StrategyLearner(verbose=False, impact=impact, commission=0)  # constructor
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
                                     commission=0,
                                     impact=impact)

        df_sl = df_sl / df_sl.iloc[0]

        print(f"Cumulative Return when impact = {impact} : {np.round_(df_sl[-1] / df_sl[0], 6) - 1}")
        print(f"Number of trades when impact = {impact} : {np.count_nonzero(df_trades_sl)}")


