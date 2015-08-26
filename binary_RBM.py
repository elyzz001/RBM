import numpy as np

class binary_RBM(object):

	def __init__(self,n_visible=None,n_hidden=256,miniBatch=100,L2_decay=None,k=1):
		self.n_hidden=n_hidden
		self.n_visible=n_visible
		self.miniBatch=miniBatch
		self.L2_decay=L2_decay
		self.k=k
		self.W=np.random.randn(n_hidden,n_visible)
		self.hbias=np.zeros((n_hidden,1))
		self.vbias=np.zeros((n_visible,1))
		
	def fit(self,x):
		
    def sigmoid(z):
		return 1/(1+np.exp(-z))
	
	def _positivePhase(self,x):
		p=self.sigmoid(self.W.dot(x.T)+self.hbias);
		
		return hidden_states