""""""  		  	   		  	  			  		 			     			  	 
"""Assess a betting strategy.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
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
  		  	   		  	  			  		 			     			  	 
Student Name: Nick DiNapoli 		  	   		  	  			  		 			     			  	 
GT User ID: ndinapoli6  		  	   		  	  			  		 			     			  	 
GT ID: 903657316 		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
def author():  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    :return: The GT username of the student  		  	   		  	  			  		 			     			  	 
    :rtype: str  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    return "ndinapoli6"  # replace tb34 with your Georgia Tech username.
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
def gtid():  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    :return: The GT ID of the student  		  	   		  	  			  		 			     			  	 
    :rtype: int  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    return 903657316  # replace with your GT ID number
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
def get_spin_result(win_prob):  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    :param win_prob: The probability of winning  		  	   		  	  			  		 			     			  	 
    :type win_prob: float  		  	   		  	  			  		 			     			  	 
    :return: The result of the spin.  		  	   		  	  			  		 			     			  	 
    :rtype: bool  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    result = False  		  	   		  	  			  		 			     			  	 
    if np.random.random() <= win_prob:  		  	   		  	  			  		 			     			  	 
        result = True  		  	   		  	  			  		 			     			  	 
    return result  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 

# Experiment 1
def test_code():
    """
    Method to test your code
    """
    win_prob = 18/38  # set appropriately to the probability of a win
    # np.random.seed(gtid())  # do this only once
    # print(get_spin_result(win_prob))  # test the roulette spin

    winnings = np.zeros((1,), dtype=int)
    episode_winnings = 0
    while episode_winnings < 80:
        won = False
        bet_amount = 1
        while not won:
            # wager bet_amount on black
            won = get_spin_result(win_prob)
            if won == True:
                episode_winnings = episode_winnings + bet_amount
            else:
                episode_winnings = episode_winnings - bet_amount
                bet_amount = bet_amount * 2

            winnings = np.append(winnings, episode_winnings)

    # winnings.extend((1000 - len(winnings)) * [80])
    winnings = np.concatenate((winnings, np.full((1000 - len(winnings)), 80)))
    return winnings


if __name__ == "__main__":
    # 3.2 Figure 1
    num_episodes = 10
    # winnings = np.empty(10)
    for episode in range(num_episodes):
        plt.plot(test_code(), label = "ep {}".format(str(episode)))

    plt.xlim([0, 300])
    plt.ylim([-256, 100])
    plt.xlabel("Wager/bet number")
    plt.ylabel("Winnings")
    plt.title("10 Episodes Winnings")
    plt.legend()
    # plt.show()
    plt.savefig('./images/Figure 1.png')
    plt.close()

    # 3.2 Figure 2
    num_episodes = 1000
    # episodes are rows
    winnings_array = np.empty(shape=(num_episodes, 1000))
    for episode in range(num_episodes):
        winnings_array[episode] = test_code()

    df_winnings = pd.DataFrame(winnings_array)

    # number of experiments that won $80
    # print(np.count_nonzero(winnings_array[:, 999] == 80))

    # print(winnings_array)
    plt.plot(df_winnings.mean(axis=0), color='b', label='mean winnings')
    plt.plot(df_winnings.mean(axis=0) + df_winnings.std(axis=0), color='g', linestyle='dashed',
             label='mean + 1 std dev')
    plt.plot(df_winnings.mean(axis=0) - df_winnings.std(axis=0), color='r', linestyle='dashed',
             label='mean - 1 std dev')
    # print(df_winnings.std(axis=0))
    plt.xlim([0, 300])
    plt.ylim([-256, 100])
    plt.xlabel("Wager/bet number")
    plt.ylabel("Mean winnings")
    plt.title("Mean Winnings of 1000 Episodes")
    plt.legend()
    # plt.show()
    plt.savefig('./images/Figure 2.png')
    plt.close()

    # now for the median
    plt.plot(df_winnings.median(axis=0), color='b', label='median winnings')
    plt.plot(df_winnings.median(axis=0) + df_winnings.std(axis=0), color='g', linestyle='dashed',
             label='median + 1 std dev')
    plt.plot(df_winnings.median(axis=0) - df_winnings.std(axis=0), color='r', linestyle='dashed',
             label='median - 1 std dev')
    # print(df_winnings.std(axis=0))
    plt.xlim([0, 300])
    plt.ylim([-256, 100])
    plt.xlabel("Wager/bet number")
    plt.ylabel("Median winnings")
    plt.title("Median Winnings of 1000 Episodes")
    plt.legend()
    # plt.show()
    plt.savefig('./images/Figure 3.png')
    plt.close()

"""=================================================================================="""

# Experiment 2
def test_code2():
    """
    Method to test your code
    """
    win_prob = 18/38  # set appropriately to the probability of a win
    # np.random.seed(gtid())  # do this only once
    # print(get_spin_result(win_prob))  # test the roulette spin

    winnings = np.zeros((1,), dtype=int)
    episode_winnings = 0
    while episode_winnings < 80 and episode_winnings > -256:
        won = False
        bet_amount = 1
        if bet_amount > episode_winnings + 256:
            bet_amount = episode_winnings + 256
        while not won:
            # wager bet_amount on black
            won = get_spin_result(win_prob)
            if won == True:
                episode_winnings = episode_winnings + bet_amount
            else:
                episode_winnings = episode_winnings - bet_amount
                bet_amount = bet_amount * 2

            winnings = np.append(winnings, episode_winnings)

    if episode_winnings > 0:
        winnings = np.concatenate((winnings, np.full((1000 - len(winnings)), 80)))
    else:
        winnings = np.concatenate((winnings, np.full((1000 - len(winnings)), -256)))

    return winnings

if __name__ == "__main__":
    # 3.3 Figure 4 & 5
    num_episodes = 1000
    # episodes are rows
    winnings_array = np.empty(shape=(num_episodes, 1000))
    for episode in range(num_episodes):
        winnings_array[episode] = test_code2()

    # number of experiments that won $80
    # print(np.count_nonzero(winnings_array[:, 999] == 80))

    df_winnings = pd.DataFrame(winnings_array)
    # print(winnings_array)
    plt.plot(df_winnings.mean(axis=0), color='b', label='mean winnings')
    plt.plot(df_winnings.mean(axis=0) + df_winnings.std(axis=0), color='g', linestyle='dashed',
             label='mean + 1 std dev')
    plt.plot(df_winnings.mean(axis=0) - df_winnings.std(axis=0), color='r', linestyle='dashed',
             label='mean - 1 std dev')
    # print(df_winnings)
    plt.xlim([0, 300])
    plt.ylim([-256, 100])
    plt.xlabel("Wager/bet number")
    plt.ylabel("Mean winnings")
    plt.title("Mean Winnings of 1000 Episodes using Realistic Strategy")
    plt.legend()
    # plt.show()
    plt.savefig('./images/Figure 4.png')
    plt.close()

    # now for the median
    plt.plot(df_winnings.median(axis=0), color='b', label='median winnings')
    plt.plot(df_winnings.median(axis=0) + df_winnings.std(axis=0), color='g', linestyle='dashed',
             label='median + 1 std dev')
    plt.plot(df_winnings.median(axis=0) - df_winnings.std(axis=0), color='r', linestyle='dashed',
             label='median - 1 std dev')
    # print(df_winnings.std(axis=0))
    plt.xlim([0, 300])
    plt.ylim([-256, 100])
    plt.xlabel("Wager/bet number")
    plt.ylabel("Median winnings")
    plt.title("Median Winnings of 1000 Episodes using Realistic Strategy")
    plt.legend()
    # plt.show()
    plt.savefig('./images/Figure 5.png')
    plt.close()