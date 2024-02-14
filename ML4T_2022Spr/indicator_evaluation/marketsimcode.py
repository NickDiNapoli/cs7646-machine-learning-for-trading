""""""  		  	   		  	  			  		 			     			  	 
"""			     			  	 
Student Name: Nick DiNapoli 		  	   		  	  			  		 			     			  	 
GT User ID: ndinapoli6 		  	   		  	  			  		 			     			  	 
GT ID: 903657316  		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import datetime as dt  		  	   		  	  			  		 			     			  	 
import os  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import numpy as np  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data, plot_data
import TheoreticallyOptimalStrategy as tos

def author():
    return 'ndinapoli6'

def compute_portvals(
    orders_file,
    symbol='JPM',
    sd=dt.datetime(2008, 1, 1),
    ed=dt.datetime(2009, 12, 31),
    start_val=100000,
    commission=0.0,
    impact=0.0,
):  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    Computes the portfolio values.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    :param orders_file: dataframe of orders indexed by date 		  	   		  	  			  		 			     			  	 
    :type orders_file: DataFrame 		  	   		  	  			  		 			     			  	 
    :param start_val: The starting value of the portfolio  		  	   		  	  			  		 			     			  	 
    :type start_val: int  		  	   		  	  			  		 			     			  	 
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		  	  			  		 			     			  	 
    :type commission: float  		  	   		  	  			  		 			     			  	 
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		  	  			  		 			     			  	 
    :type impact: float  		  	   		  	  			  		 			     			  	 
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		  	  			  		 			     			  	 
    :rtype: pandas.DataFrame  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 

    orders_df = orders_file
    orders_df.columns = ['Trades']
    # print(orders_df)

    # Daily prices of equities bought/sold in order sheet
    df_prices = get_data([symbol], pd.date_range(sd, ed))
    df_prices = df_prices.drop(['SPY'], axis=1)
    df_prices['Cash'] = np.ones(shape=(df_prices.shape[0]))
    # print(df_prices)

    # Changes in number of shares
    df_trades = pd.DataFrame(np.zeros(shape=(df_prices.shape[0], df_prices.shape[1])),
                             columns=df_prices.columns,
                             index=df_prices.index)
    # print(df_trades)

    # print(orders_df)
    for index, row in orders_df.iterrows():
        # print(index)
        if np.sign(row['Trades']) == 1:
            df_trades.loc[index, symbol] += row['Trades']
            df_trades.loc[index, 'Cash'] += -1 * (row['Trades'] * (df_prices.loc[index, symbol]*(1+impact))) -commission

        if np.sign(row['Trades']) == -1:
            df_trades.loc[index, symbol] += row['Trades']
            df_trades.loc[index, 'Cash'] += -1 * (row['Trades'] * (df_prices.loc[index, symbol]*(1-impact))) -commission


    # print(df_trades)

    df_holdings = df_trades.copy()
    df_holdings.loc[df_trades.index.values[0], 'Cash'] += start_val

    # print(df_holdings)

    for i in range(1, df_holdings.shape[0]):
        df_holdings.iloc[i] += df_holdings.iloc[i-1]

    #pd.set_option("display.max_rows", None, "display.max_columns", None)
    # print(df_holdings)


    # dollar value of each asset on each day
    df_value = df_holdings.mul(df_prices, axis='index')
    # print(df_value)

    portval = df_value.sum(axis=1)
    # print(portval)

    '''
    print(orders_df)
    print('----------------------------------------------')
    print(df_prices)
    print('----------------------------------------------')
    print(df_trades)
    print('----------------------------------------------')
    print(df_holdings)
    print('----------------------------------------------')
    print(df_value)
    print('----------------------------------------------')
    print(portval)
    '''
    # rv = pd.DataFrame(index=portvals.index, data=portvals.values)
    return portval

  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
if __name__ == "__main__":  		  	   		  	  			  		 			     			  	 

    df_bench = compute_portvals(orders_file=tos.testPolicy())
    df_bench = df_bench / df_bench.iloc[0]
    df_bench.plot(title="Daily Portfolio Value of TOS and Benchmark", fontsize=12)
    plt.xlabel("Date")
    plt.ylabel("Value (normalized)")
    # plt.show()
    # plt.savefig('./images/Figure 1.png')
    # plt.close()
