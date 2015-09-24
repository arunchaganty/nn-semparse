"""A vanilla RNN IRW model."""
import numpy
import random
import sys
import theano
from theano.ifelse import ifelse
from theano import tensor as T
from spec import IRWSpec

class RNNIRWSpec(IRWSpec):
  """Standard RNN with IRW.

  Separate matrices control contribution from read and write vectors.

  Parameters are named after functions in the irw-doc.pdf file.

  Note: this has yet to work on anything...
  it just learns to ignore input and write, so basically a language model.
  """
  def create_vars(self):
    # Initial state
    self.h0 = theano.shared(
        name='h0', 
        value=0.2 * numpy.random.uniform(-1.0, 1.0, self.nh).astype(theano.config.floatX))

    # Embedding to hidden layer
    self.u_r = theano.shared(
        name='u_r',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.de_in, self.nh)).astype(theano.config.floatX))
    self.u_w = theano.shared(
        name='u_w',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.de_out, self.nh)).astype(theano.config.floatX))
    self.u_h = theano.shared(
        name='u_h',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))

    # Logistic regression node to choose read or write
    self.w_b = theano.shared(
        name='w_b', 
        value=0.2 * numpy.random.uniform(-1.0, 1.0, self.nh).astype(theano.config.floatX))

    # Softmax layer to choose what to write
    self.w_c = theano.shared(
        name='w_c', 
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nw_out)).astype(theano.config.floatX))

    # Params
    self.params = [
        self.h0,
        self.u_r, self.u_w, self.u_h,
        self.w_b,
        self.w_c,
    ]

  def f_rnn(self, r_t, w_t, h_prev):
    r_emb = self.f_embedding(r_t)
    w_emb = self.f_embedding(w_t)
    return T.nnet.sigmoid(
        T.dot(h_prev, self.u_h) + T.dot(r_emb, self.u_r) + T.dot(w_emb, self.u_w))

  def f_p_read(self, h):
    return T.nnet.sigmoid(T.dot(h, self.w_b))

  def f_dist_write(self, h):
    return T.nnet.softmax(T.dot(h, self.w_c))[0]

