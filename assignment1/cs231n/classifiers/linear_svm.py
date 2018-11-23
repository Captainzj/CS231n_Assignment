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
  - reg: (float) regularization strength 正则化强度

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  """
  example:
  #W.shape == (D,C) == (3073, 10)      D:dimension   C:class
  #X_dev.shape == (N,D) == (500,3073)     N:number 
  #y_dev.shape == (N,) == (500,)
  #grad.shape == (3073, 10) = W.shape
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero #

  # compute the loss and the gradient
  num_classes = W.shape[1] #10
  num_train = X.shape[0]  #500
  loss = 0.0
  for i in range(num_train):  #[0,500)

    scores = X[i].dot(W)   #矩阵乘法  (1,3073)*(3073,10)
    correct_class_score = scores[y[i]] #S_yi  该图像在正确标签上的得分

    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note: delta = 1  
      if margin > 0: #在真实标签上的模型得分与该分类模型上得分差距不满足大于delta时计算损失    margin:L_i中的子项   
        loss += margin

        # Compute gradients (one inner and one outer sum)
        # Wonderfully compact and hard to read
        dW[:, y[i]] -= X[i, :].T # this is really a sum over j != y_i
        dW[:, j] += X[i, :].T # sums each contribution of the x_i's

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 *reg * np.sum(W * W)  #np.sum(W * W)对所有参数进行逐元素的平方惩罚
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
  dW = np.zeros(W.shape) # initialize the gradient as zero  #dW.shape==(3073,500)

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  num_classes=W.shape[1]
  num_train=X.shape[0]

  scores=X.dot(W) #(500,10) = (500,3073)*(3073,10)
  scores_correct = scores[np.arange(num_train), y]  #(500,) has to reshape, or will be ValueError   scores_correct[i]=scores[i,y[i]]
  scores_correct=np.reshape(scores_correct,(num_train,-1))  #(500,1)  = (500,500*1/500)

  margins=scores-scores_correct+1 #delta=1  #scores.shape=(500,10)
  margins=np.maximum(0,margins)
  margins[np.arange(num_train),y]=0#在计算loss时不把真实标签对应的得分之差delta计进去，即公式中j!=yi
  loss=np.sum(margins)/num_train
  loss += 0.5 * reg * np.sum(W * W)

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
  # compute the gradient
  margins[margins > 0] = 1  #margins中大于0的元素，数值赋为1;其余数值不变   shape==(500,10)
  row_sum = np.sum(margins, axis=1)                  # 1 by N  (1行N列)
  margins[np.arange(num_train), y] = -row_sum        #margins[np.arange(num_train), y] 初始值值为0   shape==(500,)
  #print(margins)  ##necessary to understand  
  dW += np.dot(X.T, margins)/num_train + reg * W     # D by C   dW.shape==(3073，10)  X.T.shape==(3073,500)  margins.shape==(500,10)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
