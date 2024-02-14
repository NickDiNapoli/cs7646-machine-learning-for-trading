""""""
import numpy as np

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
import random  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import pandas as pd  		  	   		  	  			  		 			     			  	 
import util as ut
import indicators as indc
import QLearner as ql
import marketsimcode as mkt
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
class StrategyLearner(object):  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
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
        self.qlearner = ql.QLearner(num_states=10000,
                               num_actions=3,
                               alpha=0.2,
                               gamma=0.9,
                               rar=0.5,
                               radr=0.99,
                               dyna=0,
                               verbose=False)

    def author(self):
        return 'ndinapoli6'

    def discretize(self, ind_vals, symbol):
        #print('indicator values:')
        indicator_values = ind_vals.copy()[symbol].tolist()
        indicator_values.sort()
        #print(indicator_values)
        #thresholds = [min(indicator_values)]
        thresholds = []
        # step = indicator_values.shape[0] / 10
        step = ind_vals.shape[0] / 10
        #indicator_values.sort_values(by=symbol)
        thresholds = [indicator_values[int(i * step)] for i in range(10)]
        #for i in range(10):
            # thresholds.append(indicator_values.iloc[int((i+1)*step) - 1])
            # thresholds.append(indicator_values[int((i + 1) * step) - 1])
            #thresholds.append(indicator_values[int(i * step)])
        # print(ind_vals.to_numpy(), np.asarray(thresholds))
        # print(thresholds)
        discrete = np.digitize(ind_vals.to_numpy(), np.asarray(thresholds))

        return discrete

  		  	   		  	  			  		 			     			  	 
    # this method should create a QLearner, and train it for trading  		  	   		  	  			  		 			     			  	 
    def add_evidence(  		  	   		  	  			  		 			     			  	 
        self,  		  	   		  	  			  		 			     			  	 
        symbol,
        sd=dt.datetime(2008, 1, 1),  		  	   		  	  			  		 			     			  	 
        ed=dt.datetime(2009, 12, 31),
        sv=100000,
    ):  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
        Trains your strategy learner over a given time frame.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
        :param symbol: The stock symbol to train on  		  	   		  	  			  		 			     			  	 
        :type symbol: str  		  	   		  	  			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  	  			  		 			     			  	 
        :type sd: datetime  		  	   		  	  			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  	  			  		 			     			  	 
        :type ed: datetime  		  	   		  	  			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		  	  			  		 			     			  	 
        :type sv: int  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
        # add your code to do learning here
        # example usage of the old backward compatible util function
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols

        prices = prices / prices.iloc[0]

        # print(prices.shape[0])
        features_array = np.zeros((prices.shape[0], 4))

        sma_values = indc.SMA(symbol, sd=sd, ed=ed, n=50).fillna(method='backfill')
        mom_values = indc.momentum(symbol, sd=sd, ed=ed, n=20).fillna(method='backfill')
        b_values = indc.perc_B(symbol, sd=sd, ed=ed, n=20).fillna(method='backfill')
        tsi_values = indc.TSI(symbol, sd=sd, ed=ed).fillna(method='backfill')
        # construct feature array and states
        features_array[:, 0] = np.squeeze(self.discretize(sma_values, symbol)) - 1
        features_array[:, 1] = np.squeeze(self.discretize(mom_values, symbol)) - 1
        features_array[:, 2] = np.squeeze(self.discretize(b_values, symbol)) - 1
        features_array[:, 3] = np.squeeze(self.discretize(tsi_values, symbol)) - 1
        #print(b_values)

        states = []
        for row in features_array:
            temp = ''
            for d in row:
                temp += str(int(d))

            states.append(int(temp))
        # df_states = pd.DataFrame(states, index=prices.index, columns=[symbol])

        # train Q Learner

        count = 0
        converged = False
        df_trades = pd.DataFrame(np.zeros((prices.shape[0])), index=prices.index, columns=[symbol])
        #print(prices.iloc[-1])
        baseline_return = prices.iloc[-1] / prices.iloc[0] - 1
        while not converged:
            holdings = 0
            a = 0
            self.qlearner.querysetstate(states[0])
            #for index, row in prices.iterrows():
            for day, state in enumerate(states):
                if day == len(states) - 1: break

                # calculate returns/reward
                if a != 0:
                    daily_rets_temp = mkt.compute_portvals(orders_file=df_trades,
                                             sd=sd,
                                             ed=ed,
                                             symbol=symbol,
                                             start_val=sv,
                                             commission=9.95,
                                             impact=0.005)

                    daily_rets_temp[1:] = (daily_rets_temp[1:] / daily_rets_temp[:-1].values) - 1
                    daily_rets_temp.iloc[0] = 0
                    r = daily_rets_temp.iloc[day + 1]
                else:
                    r = 0

                if day == 0:
                    r = 0

                a = self.qlearner.query(s_prime=state, r=r)
                # print(a)

                if a == 1:
                    if holdings == -1000:
                        df_trades.iloc[day] = 2000
                        holdings = 1000
                    if holdings == 0:
                        df_trades.iloc[day] = 1000
                        holdings = 1000

                if a == 2:
                    if holdings == 1000:
                        df_trades.iloc[day] = -2000
                        holdings = -1000
                    if holdings == 0:
                        df_trades.iloc[day] = -1000
                        holdings = -1000

            #if daily_rets_temp.iloc[-1] / daily_rets_temp.iloc[0] - 1 == baseline_return:
                #converged = True

            #baseline_return = daily_rets_temp.iloc[-1] / daily_rets_temp.iloc[0] - 1
            count += 1
            if count == 2: converged = True

        return(df_trades)

  		  	   		  	  			  		 			     			  	 
    # this method should use the existing policy and test it against new data  		  	   		  	  			  		 			     			  	 
    def testPolicy(  		  	   		  	  			  		 			     			  	 
        self,  		  	   		  	  			  		 			     			  	 
        symbol,
        sd=dt.datetime(2010, 1, 1),
        ed=dt.datetime(2011, 12, 31),
        sv=100000,
    ):  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
        Tests your learner using data outside of the training data  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
        :param symbol: The stock symbol that you trained on on  		  	   		  	  			  		 			     			  	 
        :type symbol: str  		  	   		  	  			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  	  			  		 			     			  	 
        :type sd: datetime  		  	   		  	  			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  	  			  		 			     			  	 
        :type ed: datetime  		  	   		  	  			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		  	  			  		 			     			  	 
        :type sv: int  		  	   		  	  			  		 			     			  	 
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		  	  			  		 			     			  	 
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		  	  			  		 			     			  	 
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		  	  			  		 			     			  	 
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		  	  			  		 			     			  	 
        :rtype: pandas.DataFrame  		  	   		  	  			  		 			     			  	 
        """

        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices = prices / prices.iloc[0]

        # compute states
        features_array_test = np.zeros((prices.shape[0], 4))

        sma_values_test = indc.SMA(symbol, sd=sd, ed=ed, n=50).fillna(method='backfill')
        mom_values_test = indc.momentum(symbol, sd=sd, ed=ed, n=20).fillna(method='backfill')
        b_values_test = indc.perc_B(symbol, sd=sd, ed=ed, n=20).fillna(method='backfill')
        tsi_values_test = indc.TSI(symbol, sd=sd, ed=ed).fillna(method='backfill')
        #print(b_values_test)
        # construct feature array and states
        features_array_test[:, 0] = np.squeeze(self.discretize(sma_values_test, symbol)) - 1
        features_array_test[:, 1] = np.squeeze(self.discretize(mom_values_test, symbol)) - 1
        features_array_test[:, 2] = np.squeeze(self.discretize(b_values_test, symbol)) - 1
        features_array_test[:, 3] = np.squeeze(self.discretize(tsi_values_test, symbol)) - 1

        states = []
        for row in features_array_test:
            temp = ''
            for d in row:
                temp += str(int(d))

            states.append(int(temp))

        df_trades = pd.DataFrame(np.zeros((prices.shape[0])), index=prices.index, columns=[symbol])
        self.qlearner.querysetstate(states[0])
        holdings = 0
        for day, state in enumerate(states):

            a = self.qlearner.querysetstate(state)

            if a == 1:
                if holdings == -1000:
                    df_trades.iloc[day] = 2000
                    holdings = 1000
                if holdings == 0:
                    df_trades.iloc[day] = 1000
                    holdings = 1000

            if a == 2:
                if holdings == 1000:
                    df_trades.iloc[day] = -2000
                    holdings = -1000
                if holdings == 0:
                    df_trades.iloc[day] = -1000
                    holdings = -1000

        return df_trades
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
if __name__ == "__main__":  		  	   		  	  			  		 			     			  	 
    print("One does not simply think up a strategy")

