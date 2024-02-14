""""""  		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
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
"""  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import math  		  	   		  	  			  		 			     			  	 
import sys  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

  		  	   		  	  			  		 			     			  	 
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it
  		  	   		  	  			  		 			     			  	 
if __name__ == "__main__":  		  	   		  	  			  		 			     			  	 
    if len(sys.argv) != 2:  		  	   		  	  			  		 			     			  	 
        print("Usage: python testlearner.py <filename>")  		  	   		  	  			  		 			     			  	 
        sys.exit(1)  		  	   		  	  			  		 			     			  	 
    inf = open(sys.argv[1])
    # print(sys.argv[1])
    if sys.argv[1] == 'Data/Istanbul.csv':
        data = pd.read_csv('Data/Istanbul.csv')
        data = data.values[:, 1:].astype('float64')
    else:
        data = np.array(
            [list(map(float, s.strip().split(","))) for s in inf.readlines()]
        )
  		  	   		  	  			  		 			     			  	 
    # compute how much of the data is training and testing  		  	   		  	  			  		 			     			  	 
    train_rows = int(0.6 * data.shape[0])  		  	   		  	  			  		 			     			  	 
    test_rows = data.shape[0] - train_rows  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    # separate out training and testing data  		  	   		  	  			  		 			     			  	 
    train_x = data[:train_rows, 0:-1]  		  	   		  	  			  		 			     			  	 
    train_y = data[:train_rows, -1]  		  	   		  	  			  		 			     			  	 
    test_x = data[train_rows:, 0:-1]  		  	   		  	  			  		 			     			  	 
    test_y = data[train_rows:, -1]  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    print(f"{test_x.shape}")  		  	   		  	  			  		 			     			  	 
    print(f"{test_y.shape}")  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    # create a learner and train it
    '''
    learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner  		  	   		  	  			  		 			     			  	 
    learner.add_evidence(train_x, train_y)  # train it  		  	   		  	  			  		 			     			  	 
    print(learner.author())  		  	   		  	  			  		 			     			  	 
  	'''


    # DTLearner
    learner = dt.DTLearner(leaf_size=1, verbose=False)  # create a DTLearner
    learner.add_evidence(train_x, train_y)  # train it
    print(learner.author())


    '''
    # RTLearner
    learner = rt.RTLearner(leaf_size=1, verbose=False)  # create a RTLearner
    learner.add_evidence(train_x, train_y)  # train it
    print(learner.author())
    '''

    '''
    # BagLearner
    learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": 1}, bags=20, boost=False, verbose=False)
    learner.add_evidence(train_x, train_y)
    print(learner.author())
    # Y = learner.query(Xtest)
    '''

    '''
    # InsaneLearner
    learner = it.InsaneLearner(verbose=False)  # constructor
    learner.add_evidence(train_x, train_y)  # training step
    print(learner.author())
    #Y = learner.query(Xtest)  # query
    '''

    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0,1]}")

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0,1]}")

    # Experiment 1 evaluation

    # in-sample
    RMSE_in = np.empty(shape=26)
    RMSE_out = np.empty(shape=26)
    for leaf_size in range(26):
        learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)  # create a DTLearner
        learner.add_evidence(train_x, train_y)

        pred_y = learner.query(train_x)  # get the predictions
        RMSE_in[leaf_size] = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])

        pred_y_out = learner.query(test_x)  # get the predictions
        RMSE_out[leaf_size] = math.sqrt(((test_y - pred_y_out) ** 2).sum() / test_y.shape[0])

    # print(f"min rmse: {np.argmin(RMSE_out)}")
    plt.figure()
    plt.plot(np.arange(1, 27), RMSE_in, label="in-sample")
    plt.plot(np.arange(1, 27), RMSE_out, label="out-of-sample")
    plt.legend()
    plt.title("Error as leaf size varies", fontsize=12)
    plt.xlabel("leaf size")
    plt.ylabel("RMSE")
    plt.savefig('./images/Figure 1.png')
    plt.show()

    # Experiment 2 and 3 evaluation

    # in-sample
    RMSE_in_2 = np.empty(shape=26)
    RMSE_out_2 = np.empty(shape=26)
    RMSE_in_rt = np.empty(shape=26)
    RMSE_out_rt = np.empty(shape=26)

    MAE_in_dt = np.empty(shape=26)
    MAE_out_dt = np.empty(shape=26)
    MAE_in_rt = np.empty(shape=26)
    MAE_out_rt = np.empty(shape=26)

    time_dt = np.empty(shape=26)
    time_rt = np.empty(shape=26)

    for leaf_size in range(26):
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": leaf_size}, bags=50, boost=False, verbose=False)
        start_dt = time.time()
        learner.add_evidence(train_x, train_y)
        end_dt = time.time()
        time_dt[leaf_size] = end_dt - start_dt

        learner_rt = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": leaf_size}, bags=50, boost=False, verbose=False)
        start_rt = time.time()
        learner_rt.add_evidence(train_x, train_y)
        end_rt = time.time()
        time_rt[leaf_size] = end_rt - start_rt

        pred_y = learner.query(train_x)  # get the predictions
        RMSE_in_2[leaf_size] = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        MAE_in_dt[leaf_size] = np.abs(train_y - pred_y).sum() / train_y.shape[0]

        pred_y_out = learner.query(test_x)  # get the predictions
        RMSE_out_2[leaf_size] = math.sqrt(((test_y - pred_y_out) ** 2).sum() / test_y.shape[0])
        MAE_out_dt[leaf_size] = np.abs(test_y - pred_y_out).sum() / test_y.shape[0]

        # random trees
        pred_y_rt = learner_rt.query(train_x)  # get the predictions
        RMSE_in_rt[leaf_size] = math.sqrt(((train_y - pred_y_rt) ** 2).sum() / train_y.shape[0])
        MAE_in_rt[leaf_size] = np.abs(train_y - pred_y_rt).sum() / train_y.shape[0]

        pred_y_out_rt = learner_rt.query(test_x)  # get the predictions
        RMSE_out_rt[leaf_size] = math.sqrt(((test_y - pred_y_out_rt) ** 2).sum() / test_y.shape[0])
        MAE_out_rt[leaf_size] = np.abs(test_y - pred_y_out_rt).sum() / test_y.shape[0]

    print(f"min rmse: {np.argmin(RMSE_out)}")
    print(f"min rmse dt: {np.argmin(RMSE_out_2)}")
    print(f"min rmse rt: {np.argmin(RMSE_out_rt)}")
    plt.figure()
    plt.plot(np.arange(1, 27), RMSE_in, label="single tree in-sample")
    plt.plot(np.arange(1, 27), RMSE_out, label="single tree out-of-sample")
    plt.plot(np.arange(1, 27), RMSE_in_2, label="50 bag ensemble in-sample")
    plt.plot(np.arange(1, 27), RMSE_out_2, label="50 bag ensemble out-of-sample")
    plt.legend()
    plt.title("Error as leaf size varies for single tree vs. an ensemble with bagging", fontsize=12)
    plt.xlabel("leaf size")
    plt.ylabel("RMSE")
    plt.savefig('./images/Figure 2.png')
    plt.show()

    plt.figure()
    plt.plot(np.arange(1, 27), RMSE_in, label="single tree in-sample")
    plt.plot(np.arange(1, 27), RMSE_out, label="single tree out-of-sample")
    plt.plot(np.arange(1, 27), RMSE_in_rt, label="50 bag random tree ensemble in-sample")
    plt.plot(np.arange(1, 27), RMSE_out_rt, label="50 bag random tree ensemble out-of-sample")
    plt.legend()
    plt.title("Error as leaf size varies for single tree vs. an ensemble of random trees with bagging", fontsize=10)
    plt.xlabel("leaf size")
    plt.ylabel("RMSE")
    plt.savefig('./images/Figure 3.png')
    plt.show()

    # Exp 3 figures

    plt.figure()
    plt.plot(np.arange(1, 27), MAE_in_dt, label="normal trees ensemble in-sample")
    plt.plot(np.arange(1, 27), MAE_out_dt, label="normal trees ensemble out-of-sample")
    plt.plot(np.arange(1, 27), MAE_in_rt, label="random trees ensemble in-sample")
    plt.plot(np.arange(1, 27), MAE_out_rt, label="random trees ensemble out-of-sample")
    plt.legend()
    plt.title("MAE as leaf size varies for normal tree ensemble vs. random tree ensemble with bagging", fontsize=10)
    plt.xlabel("leaf size")
    plt.ylabel("MAE")
    plt.savefig('./images/Figure 4.png')
    plt.show()

    plt.figure()
    plt.plot(np.arange(1, 27), time_dt, label="50 normal trees ensemble")
    plt.plot(np.arange(1, 27), time_rt, label="50 random trees ensemble")
    plt.legend()
    plt.title("Time to train normal tree ensemble vs. random tree ensemble with bagging", fontsize=10)
    plt.xlabel("leaf size")
    plt.ylabel("Time")
    plt.savefig('./images/Figure 5.png')
    plt.show()