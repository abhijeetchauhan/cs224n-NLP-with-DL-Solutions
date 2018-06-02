#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
	"""
	Forward and backward propagation for a two-layer sigmoidal network

	Compute the forward propagation and for the cross entropy cost,
	and backward propagation for the gradients for all parameters.

	Arguments:
	data -- M x Dx matrix, where each row is a training example.
	labels -- M x Dy matrix, where each row is a one-hot vector.
	params -- Model parameters, these are unpacked for you.
	dimensions -- A tuple of input dimension, number of hidden units
				  and output dimension
	"""

	### Unpack network parameters (do not modify)
	ofs = 0
	Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

	print "forward and back "
	# print params.shape
	# print Dx*H + H*Dy + H + Dy
	# print "done"

	W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
	ofs += Dx * H
	b1 = np.reshape(params[ofs:ofs + H], (1, H))
	ofs += H
	W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
	ofs += H * Dy
	b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))
	N, D = data.shape
	### YOUR CODE HERE: forward propagation
	z1 = np.dot(data,W1)+b1 # M x H
	a1 = sigmoid(z1)  # M x H
	z2 = np.dot(a1,W2) + b2 # M x Dy
	a2 = softmax(z2) # M x Dy
	cost = -1*np.sum(np.multiply(labels,np.log(a2)))/N
	# raise NotImplementedError
	### END YOUR CODE

	### YOUR CODE HERE: backward propagation
	delta2 = (a2-labels) # M x Dy
	delta1 = sigmoid_grad(a1)*np.dot(delta2,W2.T) #  [(M x Dy) X (Dy x H) = M x H] 

	# W2 = W2 + np.dot(a1.T,delta2) # (H x M) X (M x Dy) = H x Dy
	# b2 = b2 + delta2 # M x Dy
	# W1 = W1 + np.dot(data.T,delta1) # (Dx x M) X (M x H) 
	# b1 = b1 + delta1 # M x H
	gradW1 = np.dot(data.T,delta1)/N
	gradb1 = np.sum(delta1,axis=0)[np.newaxis]/N
	gradW2 = np.dot(a1.T,delta2)/N
	gradb2 = np.sum(delta2,axis=0)[np.newaxis]/N

	if(gradW1.shape != W1.shape):
		print "Error in shape of gradW1"
	if(gradW2.shape != W2.shape):
		print "Error in shape of gradW2"
	if(gradb1.shape != b1.shape):
		print "Error in shape of gradb1"
		print gradb1.shape
		print gradb1
		print b1.shape
		print b1
		print delta1.shape
		print delta1
	if(gradb2.shape != b2.shape):
		print "Error in shape of gradb2"
		print gradb2.shape
		print b2.shape
		print delta2.shape
	# raise NotImplementedError
	### END YOUR CODE

	### Stack gradients (do not modify)
	grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
		gradW2.flatten(), gradb2.flatten()))

	return cost, grad


def sanity_check():
	"""
	Set up fake data and parameters for the neural network, and test using
	gradcheck.
	"""
	print "Running sanity check..."

	N = 20
	dimensions = [10, 5, 10]
	data = np.random.randn(N, dimensions[0])   # each row will be a datum
	labels = np.zeros((N, dimensions[2]))
	for i in xrange(N):
		labels[i, random.randint(0,dimensions[2]-1)] = 1

	params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
		dimensions[1] + 1) * dimensions[2], )

	gradcheck_naive(lambda params:
		forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
	"""
	Use this space add any additional sanity checks by running:
		python q2_neural.py
	This function will not be called by the autograder, nor will
	your additional tests be graded.
	"""
	print "Running your sanity checks..."
	### YOUR CODE HERE
	raise NotImplementedError
	### END YOUR CODE


if __name__ == "__main__":
	sanity_check()
	# your_sanity_checks()
