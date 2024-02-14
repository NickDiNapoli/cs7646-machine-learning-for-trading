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
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
class BagLearner(object):
    """  		  	   		  	  			  		 			     			  	 
    This is a Decision Tree Learner.
  		  	   		  	  			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  	  			  		 			     			  	 
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		  	  			  		 			     			  	 
    :type verbose: bool  		  	   		  	  			  		 			     			  	 
    """
    def __init__(self, learner, kwargs, bags, boost, verbose=False):
        """  		  	   		  	  			  		 			     			  	 
        Constructor method  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
        self.learner = learner
        self.kwargs = kwargs
        # self.leaf_size = kwargs.get('leaf_size')
        self.bags = bags
        self.boost = boost
  		  	   		  	  			  		 			     			  	 
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

        # train an arbitrary learner using bagging
        learners = []
        n = data_x.shape[0]
        # kwargs = {"k": 10}
        for i in range(self.bags):
            # inds = np.random.randint(n, size=n)
            # learners.append(self.learner(**self.kwargs).add_evidence(data_x[inds], data_y[inds]))
            learners.append(self.learner(**self.kwargs)) # .add_evidence(data_x[inds], data_y[inds]))

        for l in learners:
            inds = np.random.randint(n, size=n)
            l.add_evidence(data_x[inds], data_y[inds])

        self.learners = learners
        # print(type(learners[0]))
        return learners

    def query(self, points):  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
        Estimate a set of test points given the model we built.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		  	  			  		 			     			  	 
        :type points: numpy.ndarray  		  	   		  	  			  		 			     			  	 
        :return: The predicted result of the input data according to the trained model  		  	   		  	  			  		 			     			  	 
        :rtype: numpy.ndarray  		  	   		  	  			  		 			     			  	 
        """

        Ypred = np.empty(shape=(self.bags, points.shape[0]))
        for i, learner_trained in enumerate(self.learners):
            # print(type(learner_trained))
            Ypred[i] = learner_trained.query(points)

        # print(Ypred.shape)
        Ypred = Ypred.mean(axis=0)
        # print(Ypred.shape)
        # print(Ypred.shape[0] == points.shape[0])
        return Ypred
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
if __name__ == "__main__":  		  	   		  	  			  		 			     			  	 
    print("the secret clue is 'zzyzx'")  		  	   		  	  			  		 			     			  	 
