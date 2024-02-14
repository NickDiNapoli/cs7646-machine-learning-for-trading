""""""  		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		  	  			  		 			     			  	 
All Rights Reserved  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Template code for CS 4646/7646  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  	  			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		  	  			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  	  			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  			  		 			     			  	 
or edited.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  	  			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		  	  			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  			  		 			     			  	 
GT honor code violation.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
-----do not edit anything above this line---  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Student Name: Tucker Balch (replace with your name)  		  	   		  	  			  		 			     			  	 
GT User ID: ndinapoli6		  	   		  	  			  		 			     			  	 
GT ID: 903657316 		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import datetime as dt
import pandas as pd  		  	   		  	  			  		 			     			  	 
import util as ut
from indicators import *
import indicators as indc
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
class ManualStrategy(object):
    """  		  	   		  	  			  		 			     			  	 
    A manual strategy that can creates a trading policy using the same indicators used in StrategyLearner.
  		  	   		  	  			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  	  			  		 			     			  	 
        If verbose = False your code should not generate ANY output.  		  	   		  	  			  		 			     			  	 
    :type verbose: bool  		  	   		  	  			  		 			     			  	 
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		  	  			  		 			     			  	 
    :type impact: float  		  	   		  	  			  		 			     			  	 
    :param commission: The commission amount charged, defaults to 0.0  		  	   		  	  			  		 			     			  	 
    :type commission: float  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    # constructor  		  	   		  	  			  		 			     			  	 
    def __init__(self, verbose=False, impact=0.005, commission=9.95):
        """  		  	   		  	  			  		 			     			  	 
        Constructor method  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
        self.verbose = verbose  		  	   		  	  			  		 			     			  	 
        self.impact = impact  		  	   		  	  			  		 			     			  	 
        self.commission = commission

    def author(self):
        return 'ndinapoli6'

    # this method implements a manual trading strategy
    def testPolicy(  		  	   		  	  			  		 			     			  	 
        self,  		  	   		  	  			  		 			     			  	 
        symbol,
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        sv=100000,
    ):  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
        Tests your learner using data outside of the training data  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
        :param symbol: The stock symbol that you trained on on  		  	   		  	  			  		 			     			  	 
        :type symbol: str  		  	   		  	  			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  	  			  		 			     			  	 
        :type sd: datetime  		  	   		  	  			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 12/31/2009  		  	   		  	  			  		 			     			  	 
        :type ed: datetime  		  	   		  	  			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		  	  			  		 			     			  	 
        :type sv: int  		  	   		  	  			  		 			     			  	 
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		  	  			  		 			     			  	 
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		  	  			  		 			     			  	 
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		  	  			  		 			     			  	 
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		  	  			  		 			     			  	 
        :rtype: pandas.DataFrame  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
        # here we build a fake set of trades  		  	   		  	  			  		 			     			  	 
        # your code should return the same sort of data  		  	   		  	  			  		 			     			  	 
        dates = pd.date_range(sd, ed)  		  	   		  	  			  		 			     			  	 
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        prices = prices_all[[symbol,]]
        prices = prices / prices.iloc[0]
        trades = prices_all[[symbol,]]  # only portfolio symbols
        trades.values[:, :] = 0  # set them all to nothing
        # trades_SPY = prices_all["SPY"]  # only SPY, for comparison later

        sma = indc.SMA(symbol,
                       sd=dt.datetime(2008, 1, 1),
                       ed=dt.datetime(2009, 12, 31),
                       n=50)
        mom = indc.momentum(symbol,
                            sd=dt.datetime(2008, 1, 1),
                            ed=dt.datetime(2009, 12, 31),
                            n=20)
        b = indc.perc_B(symbol,
                        sd=dt.datetime(2008, 1, 1),
                        ed=dt.datetime(2009, 12, 31),
                        n=20)
        tsi = indc.TSI(symbol,
                        sd=dt.datetime(2008, 1, 1),
                        ed=dt.datetime(2009, 12, 31))


        i = 1
        holdings = 0
        while (i < trades.shape[0]):
            t = 0

            # SMA indicator
            if prices[symbol][i-1] < sma[symbol][i] and prices[symbol][i] > sma[symbol][i]:
                t += -1
            if prices[symbol][i-1] > sma[symbol][i] and prices[symbol][i] < sma[symbol][i]:
                t += 1

            # Momentum indicator
            if mom[symbol][i] > .2:
                t += -1
            if mom[symbol][i] < .2:
                t += 1

            # %B indicator
            if b[symbol][i] > 1.1:
                t += -1
            if b[symbol][i] < -.15:
                t += 1
            '''
            # TSI indicator
            if tsi[symbol][i-1] < 0 and tsi[symbol][i] > 0:
                t += -1
            if tsi[symbol][i-1] > 0 and tsi[symbol][i] < 0:
                t += 1
            '''
            # TSI indicator
            if tsi[symbol][i] > 20:
                t += -1
            if tsi[symbol][i] < -20:
                t += 1


            # set trade
            # weak buy
            if t == 1:
                if holdings == 0:
                    trades.values[i, :] = 1000
                    holdings = 1000
                if holdings == -1000:
                    trades.values[i, :] = 1000
                    holdings = 0
            # strong buy
            if t >= 2:
                if holdings == 0:
                    trades.values[i, :] = 1000
                    holdings = 1000
                if holdings == -1000:
                    trades.values[i, :] = 2000
                    holdings = 1000

            # weak sell
            if t == -1:
                if holdings == 0:
                    trades.values[i, :] = -1000
                    holdings = -1000
                if holdings == 1000:
                    trades.values[i, :] = -1000
                    holdings = 0

            # strong sell
            if t <= -2:
                if holdings == 0:
                    trades.values[i, :] = -1000
                    holdings = -1000
                if holdings == 1000:
                    trades.values[i, :] = -2000
                    holdings = -1000
            i += 1

        #print(trades)
        #print(trades['JPM'].to_list().count(1000.))
        #print(trades['JPM'].to_list().count(-1000.))
        #print(trades.columns)

        return trades
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
if __name__ == "__main__":  		  	   		  	  			  		 			     			  	 
    print("One does not simply think up a strategy")
