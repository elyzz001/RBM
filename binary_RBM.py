import numpy as np

class binary_RBM(object):

	def __init__(self,n_visible=None,n_hidden=256,batchSize=100,L2_decay=None,epoch=1):
		self.n_hidden=n_hidden
		self.n_visible=n_visible
		self.batchSize=batchSize
		self.L2_decay=L2_decay
		self.W=np.random.randn(n_hidden,n_visible)
		self.hbias=np.zeros(n_hidden)
		self.vbias=np.zeros(n_visible)
		self.epoches=epoches

	def fit(self,x):
		
		N=x.shape[0]
		batches,num_batches=self._batchesLists(N)

		for t in range(0,self.epoches):

			for k in range(0,num_batches):
				idx=batches[k]
				p_h=self._sigmoid(np.dot(x[idx,:],self.W.T)+self.hbias)
				h=p_h>np.random.rand(n_hidden)

				


		
		
		
		return None
	
	
	def _batchLists(self,N):
		num_batches=np.ceil(N/self.batchSize)
		batch_idx=np.tile(np.arange(0,num_batches)\
				,self.batchSize)
		batch_idx=batch_idx[0:N]
		np.random.shuffle(batch_idx)
                print batch_idx
		batch_list=[]

		for i in range(0,int(num_batches)):
			idx=np.argwhere(batch_idx==i)
			batch_list.append(idx)

		return batch_list,num_batches

	def _sigmoid(z):
		return 1/(1+np.exp(-z))


if __name__=="__main__":

