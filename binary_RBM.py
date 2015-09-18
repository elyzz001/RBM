import numpy as np

class binary_RBM(object):

	def __init__(self,n_visible=None,n_hidden=256,miniBatch=100,L2_decay=None,epoch=1):
		self.n_hidden=n_hidden
		self.n_visible=n_visible
		self.miniBatch=miniBatch
		self.L2_decay=L2_decay
		self.W=np.random.randn(n_hidden,n_visible)
		self.hbias=np.zeros((n_hidden,1))
		self.vbias=np.zeros((n_visible,1))
		self.epoch=epoch

	def fit(self,x):

		N=x.shape(0)
		for t in range(0,self.epoch):

			BatchList=self.partition_batches(N)
			for batch in BatchList:
				v=x[batch,:]
				#Positive phase



    def partition_batches(self,N):
		numBatches=np.ceil(N/self.miniBatch)
		BatchList=[]

		sortingArr=np.random.permutation(N)
		for i in range(0,numBatches,self.miniBatch):

			if
				batch=sortingArr[i:self.miniBatch*i]
			BatchList.append(batch)




    def sigmoid(z):
		return 1/(1+np.exp(-z))
