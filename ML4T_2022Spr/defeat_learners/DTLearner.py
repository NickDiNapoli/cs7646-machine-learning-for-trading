""""""  		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
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

import numpy as np


class DTLearner(object):
    """
    This is a Decision Tree Learner.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """
    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor method
        """
        self.leaf_size = leaf_size

    def author(self):
        """  		  	   		  	  			  		 			     			  	 
        :return: The GT username of the student  		  	   		  	  			  		 			     			  	 
        :rtype: str  		  	   		  	  			  		 			     			  	 
        """
        return "ndinapoli6"

    def add_evidence(self, data_x, data_y):
        """  		  	   		  	  			  		 			     			  	 
        Add training data to learner  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
        :param data_x: A set of feature values used to train the learner  		  	   		  	  			  		 			     			  	 
        :type data_x: numpy.ndarray  		  	   		  	  			  		 			     			  	 
        :param data_y: The value we are attempting to predict given the X data  		  	   		  	  			  		 			     			  	 
        :type data_y: numpy.ndarray  		  	   		  	  			  		 			     			  	 
        """

        # build a decision tree using JR Quinlan method

        # first, check two stopping conditions
        #if data_x.shape[0] == 0:
            #return np.array([-1, -1, -1, -1])

        if data_x.shape[0] > 0 and data_x.shape[0] <= self.leaf_size:
            return np.array([-1, data_y.mean(), -1, -1])

        if np.all(data_y == data_y[0]):
            return np.array([-1, data_y[0], -1, -1])

        else:
            corr_matrix = np.corrcoef(data_x, data_y, rowvar=False)
            corr = np.abs(corr_matrix[-1, :-1])
            i = np.argmax(corr)
            SplitVal = np.median(data_x[:, i])
            # SplitVal = data_x[:, i].mean()
            inds_L = np.where(data_x[:, i] <= SplitVal)[0]
            inds_R = np.where(data_x[:, i] > SplitVal)[0]

            # check corner case
            # print(inds_L.shape, inds_R.shape, inds_R.shape[0])
            # print(inds_R.shape[0] == 0)
            if inds_L.shape[0] == 0 or inds_R.shape[0] == 0:
                if inds_L.shape[0] == 0:
                    lefttree = np.array([-1, -1, -1, -1])
                    righttree = np.array([-1, data_y.mean(), -1, -1])
                if inds_R.shape[0] == 0:
                    righttree = np.array([-1, -1, -1, -1])
                    lefttree = np.array([-1, data_y.mean(), -1, -1])
            else:
                lefttree = self.add_evidence(data_x[inds_L], data_y[inds_L])
                righttree = self.add_evidence(data_x[inds_R], data_y[inds_R])

            # lefttree = self.add_evidence(data_x[inds_L], data_y[inds_L])
            if len(lefttree.shape) > 1:
                j = lefttree.shape[0]
            else:
                j = 1

            # righttree = self.add_evidence(data_x[inds_R], data_y[inds_R])
            root = np.array([i, SplitVal, 1, j+1])
            # print(root.shape, lefttree.shape, righttree.shape)

            self.DT = np.vstack((root, lefttree, righttree))
            # print(self.DT)
            # print(self.DT[:, 0])
            # print(self.DT[:, 0].astype(int))
            return self.DT


    def query(self, points):
        """  		  	   		  	  			  		 			     			  	 
        Estimate a set of test points given the model we built.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		  	  			  		 			     			  	 
        :type points: numpy.ndarray  		  	   		  	  			  		 			     			  	 
        :return: The predicted result of the input data according to the trained model  		  	   		  	  			  		 			     			  	 
        :rtype: numpy.ndarray  		  	   		  	  			  		 			     			  	 
        """
        #def eval(point, node):

        Ypred = np.empty(shape=points.shape[0])
        for point in range(points.shape[0]):
            node = 0
            factor = int(self.DT[node, 0])
            SplitVal = self.DT[node, 1]
            while(factor != -1):
                if points[point, factor] <= SplitVal:
                    node = node + int(self.DT[node, 2])
                    factor = int(self.DT[node, 0])
                    SplitVal = self.DT[node, 1]
                else:
                    node = node + int(self.DT[node, 3])
                    factor = int(self.DT[node, 0])
                    SplitVal = self.DT[node, 1]

            Ypred[point] = self.DT[node, 1]
            # print(Ypred[point])

        return Ypred
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
if __name__ == "__main__":  		  	   		  	  			  		 			     			  	 
    print("the secret clue is 'zzyzx'")  		  	   		  	  			  		 			     			  	 
