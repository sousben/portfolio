#
#
#  Vanilla RNN model
#
#

## this is a 3 layers neuron network.
## input layer: one hot vector, dim: vocab * 1
## hidden layer: RNN, hidden vector: hidden size * 1
## output layer: Softmax, vocab * 1, the probabilities distribution of each character

import numpy as np

# data I/O
data = open('input.txt', 'r').read() # should be simple plain text file

# use set() to count the vacab size
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))

# dictionary to convert char to idx, idx to char
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperWseters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# model Wseters

# TODO-1: fill in model size.
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias


## compute loss, derivative
#------------------------------------------------------------------------------
#  forward seq_length characters through the net and fetch gradient
#
#  Given
#     inputs, targets -- list of integers.
#     hprev           --  Hx1 array of initial hidden state
#     returns the loss, gradients on model Wseters, and last hidden state
#------------------------------------------------------------------------------
def lossFun(inputs, targets, hprev):

  xs, hs, ys, ps = {}, {}, {}, {}
  ## record each hidden state of
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size, 1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1

    # hidden state, using Wxh, Whh,  xs[t] and previous hidden state hs[t-1]
    hs[t] = np.tanh(np.dot(Wxh,xs[t]) + np.dot(Whh,hs[t-1]) + bh)

    ## next chars
    ys[t] = np.dot(Why, hs[t]) + by

    ## probabilities for next chars, softmax
    ps[t] = np.exp(ys[t]) / np.exp(ys[t]).sum(axis=0) #Your-code

    ## softmax (cross-entropy loss)
    loss += -np.log(ps[t][targets[t], 0])

  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y

    ## output layer doesnot use activation function, so no need to compute the derivative of error with regard to the net input
    ## of output layer.
    ## then, we could directly compute the derivative of error with regard to the weight between hidden layer and output layer.
    ## dE/dy[j]*dy[j]/dWhy[j,k] = dE/dy[j] * h[k]
    dWhy += np.dot(dy, hs[t].T)
    dby += dy

    ## backprop into h
    ## derivative of error with regard to the output of hidden layer
    ## derivative of H, come from output layer y and also come from H(t+1), the next time H
    dh = np.dot(Why.T, dy) + dhnext

    ## backprop through tanh nonlinearity
    ## derivative of error with regard to the input of hidden layer
    ## dtanh(x)/dx = 1 - tanh(x) * tanh(x)
    dhraw = (1 - hs[t] * hs[t]) * dh
    dbh += dhraw

    ## derivative of the error with regard to the weight between input layer and hidden layer
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    ## derivative of the error with regard to H(t+1)
    ## or derivative of the error of H(t-1) with regard to H(t)
    dhnext = np.dot(Whh.T, dhraw)


  for dWs in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dWs, -5, 5, out=dWs) # clip to mitigate exploding gradients

  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
## given a hidden RNN state, and a input char id, predict the coming n chars
def sample(h, seed_ix, n):
  """
  sample a sequence of integers from the model
  h is dx_gradory state, seed_ix is seed letter for first time step
  """

  ## a one-hot vector
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1

  ixes = []
  for t in range(n):
    ## hidden state, using Wxh, Wxh,  xs[t] and previous hidden state hs[t-1]
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)

    ## Output
    y = np.dot(Why, h) + by
    ## softmax

    ## probabilities for next chars, softmax
    p = np.exp(y) / np.sum(np.exp(y))
    ## sample next chars according to probability distribution
    ix = np.random.choice(range(vocab_size), p=p.ravel())

    ## update input x
    ## use the new sampled result as last input, then predict next char again.
    x = np.zeros((vocab_size, 1))
    x[ix] = 1

    ixes.append(ix)

  return ixes


## iterator counter
n = 0
## data pointer
p = 0
# variables for Adagrad

dx_grad_Wxh, dx_grad_Whh, dx_grad_Why = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
dx_grad_bh, dx_grad_by = np.zeros_like(bh), np.zeros_like(by)


smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

## main loop
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p + seq_length + 1 >= len(data) or n == 0:
    # reset RNN dx_gradory
    ## hprev is the hiddden state of RNN
    hprev = np.zeros((hidden_size, 1))
    # go from start of data
    p = 0

  inputs = [char_to_ix[ch] for ch in data[p : p + seq_length]]
  targets = [char_to_ix[ch] for ch in data[p + 1 : p + seq_length + 1]]

  # sample from the model now and then
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print('---- sample -----')
    print('----\n %s \n----' % (txt, ))

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)

  ## Adagrad + momentum
  smooth_loss = smooth_loss * 0.999 + loss * 0.001

  if n % 100 == 0:
    print('iter %d, loss: %f' % (n, smooth_loss)) # print progress


  # parameter update
  for Ws, dWs, dx_grad in zip([Wxh, Whh, Why, bh, by],
                                [dWxh, dWhh, dWhy, dbh, dby],
                                [dx_grad_Wxh, dx_grad_Whh, dx_grad_Why, dx_grad_bh, dx_grad_by]):
    dx_grad += dWs * dWs
    Ws += -learning_rate * dWs / np.sqrt(dx_grad + 1e-8) # adagrad update


  p += seq_length # move data pointer
  n += 1 # iteration counter
