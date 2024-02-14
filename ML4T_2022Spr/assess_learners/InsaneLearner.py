import numpy as np
import LinRegLearner as lrl
import BagLearner as bl
  		  	   		  	  			  		 			     			  	 
class InsaneLearner(object):

    def __init__(self, verbose=False): pass
  		  	   		  	  			  		 			     			  	 
    def author(self): return "ndinapoli6"
  		  	   		  	  			  		 			     			  	 
    def add_evidence(self, data_x, data_y):  		  	   		  	  			  		 			     			  	 

        self.learners = [bl.BagLearner(learner=lrl.LinRegLearner, kwargs = {}, bags = 20, boost = False) for i in range(20)]
        for l in self.learners:
            l.add_evidence(data_x[np.random.randint(data_x.shape[0], size=data_x.shape[0])], data_y[np.random.randint(data_x.shape[0], size=data_x.shape[0])])
        return self.learners


    def query(self, points):

        Ypred = np.empty(shape=(20, points.shape[0]))
        for i, learner_trained in enumerate(self.learners):
            Ypred[i] = learner_trained.query(points)
        Ypred = Ypred.mean(axis=0)
        return Ypred