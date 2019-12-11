# usr/bin/env python3
# -*- coding:utf8 -*-

__Author__='Florent LIANG'
__Date__='11/12/2019'

"""
1. FizzBuzz game with deep learning
2. 3 for fizz，5 for buzz，15 for fizzbuzz
"""

# declaration of library packages:
import numpy as np
import torch

class TwoLayerStruct(torch.nn.Module):
	"""define a two-layer model for training"""
	def __init__(self,data_in,data_out,hidden:int=100):
		super(TwoLayerStruct, self).__init__()
		# input to hidden layer
		self.Linear_in=torch.nn.Linear(data_in,hidden)
		# define the activiation function
		self.relu=torch.nn.ReLU()
		# hidden to output
		self.Linear_out=torch.nn.Linear(hidden,data_out)

	def forward(self,x):
		return self.Linear_out(self.relu(self.Linear_in(x)))


class GameLearning(object):
	"""docstring for GameLearning"""
	def __init__(self,num_digit:int=10):
		# number of digit for binary tranformation
		# define also the range of dataset
		self.num_digit=num_digit

		# prepare the x dateset for trainning
		self.x_train=torch.Tensor([self.to_binary(i) for i in range(101,2**num_digit)])

		# prepare the y dataset for training
		self.y_train=torch.LongTensor([self.digit_encode(i) for i in range(101,2**num_digit)])

		# prepare the x dataset for test
		self.x_test=torch.Tensor([self.to_binary(i) for i in range(0,101)])

		# define the training model
		self.model=TwoLayerStruct(self.num_digit,4)

		# define the loss function
		self.loss_fn=torch.nn.CrossEntropyLoss()

		# define the optimizer
		self.optimizer=torch.optim.SGD(self.model.parameters(),lr=0.05)

		# run
		self.train()
		self.test()

	def train(self,batch_size:int=128):
		"""train the dataset"""
		for epoch in range(15000):
			for start in range(0,len(self.x_train),batch_size):
				end=start+batch_size
				x_batch=self.x_train[start:end]
				y_batch=self.y_train[start:end] # define the batch size for check the loss

				y_pred=self.model(x_batch) # prediction of the present batch
				self.loss=self.loss_fn(y_pred,y_batch)
				print(epoch,self.loss.item())

				self.optimizer.zero_grad() # clear the weight parameter in the optimizer
				self.loss.backward() # backward probagation
				self.optimizer.step()

	def test(self):
		"""check the test dataset"""
		self.y_test=self.model(self.x_train)
		res=zip(range(0,101),self.y_test.max(1)[1].data.tolist())
		print([self.digit_decode(i,j) for i,j in res])

	def to_binary(self,i:int):
		"""transform the a given number to its binary counterpart"""
		return np.array([i>>d &1 for d in range(self.num_digit)])

	@staticmethod
	def digit_encode(i:int):
		"""define the tag for different response: train_y"""
		if i%15==0: return 3
		elif i%5==0: return 2
		elif i%3==0: return 1
		else: return 0

	@staticmethod
	def digit_decode(i:int,tag:int):
		"""tansform the tag into result"""
		return [str(i),'fizz','buzz','fizz_buzz'][tag]

if __name__ == '__main__':
	gl=GameLearning()
