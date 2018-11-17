import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):

    scores = X[i].dot(W)   #矩阵乘法
    correct_class_score = scores[y[i]] #S_yi
    '''
    print("x.shape:",X.shape) #(500, 3073) 矩阵
    print("X[i].shape:",X[i].shape) #(3073,) 向量
    print("scores's shape:",scores.shape)#(10,) 向量
    '''

    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0: #在真实标签上的模型得分与该分类模型上得分差距不满足大于delta时计算损失
        loss += margin
        # Compute gradients (one inner and one outer sum)
        # Wonderfully compact and hard to read
        dW[:, y[i]] -= X[i, :].T # this is really a sum over j != y_i
        dW[:, j] += X[i, :].T # sums each contribution of the x_i's

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  ''' 
  print("w*w:",(W*W).shape)#(3073, 10)
  print("np.sum(w*w)",(np.sum(W*W)).shape)#标量 矩阵W*W每个元素的和
  '''

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  # Gradient regularization that carries through per https://piazza.com/class/i37qi08h43qfv?cid=118
  dW += reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  scores=X.dot(W) #(500,10)
  num_classes=W.shape[1]
  num_train=X.shape[0]
  scores_correct = scores[np.arange(num_train), y]  #(500,) has to reshape, or will be ValueError
  scores_correct=np.reshape(scores_correct,(num_train,-1))
  margins=scores-scores_correct+1#delta=1
  margins=np.maximum(0,margins)
  margins[np.arange(num_train),y]=0#在计算loss时不把真实标签对应的得分之差delta计进去，即公式中j!=yi
  loss=np.sum(margins)/num_train
  loss += 0.5 * reg * np.sum(W * W)

  # compute the gradient
  margins[margins > 0] = 1
  row_sum = np.sum(margins, axis=1)                  # 1 by N
  margins[np.arange(num_train), y] = -row_sum        
  # margins[np.arange(num_train), y] -= row_sum        
  dW += np.dot(X.T, margins)/num_train + reg * W     # D by C
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
