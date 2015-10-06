"""A GRU layer."""
import numpy
import random
import sys
import theano
from theano.ifelse import ifelse
from theano import tensor as T

from rnnlayer import RNNLayer

class GRULayer(RNNLayer):
  """A GRU layer.

  Parameter names follow convention in Richard Socher's CS224D slides.
  """
  def create_vars(self, create_init_state, create_output_layer):
    # Initial state
    if create_init_state:
      self.h0 = theano.shared(
          name='h0', 
          value=0.2 * numpy.random.uniform(-1.0, 1.0, self.nh).astype(theano.config.floatX))
      init_state_params = [self.h0]
    else:
      init_state_params = []

    # Encoder hidden state updates
    self.wz = theano.shared(
        name='wz',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.de, self.nh)).astype(theano.config.floatX))
    self.uz = theano.shared(
        name='uz',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
    self.wr = theano.shared(
        name='wr',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.de, self.nh)).astype(theano.config.floatX))
    self.ur = theano.shared(
        name='ur',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
    self.w = theano.shared(
        name='w',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.de, self.nh)).astype(theano.config.floatX))
    self.u = theano.shared(
        name='u',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
    recurrence_params = [self.wz, self.uz, self.wr, self.ur, self.w, self.u]

    # Output layer
    if self.create_output_layer:
      self.w_out = theano.shared(
          name='w_out', 
          value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nw_out)).astype(theano.config.floatX))
      output_params = [self.w_out]
    else:
      output_params = []

    # Params
    self.params = init_state_params + recurrence_params + output_params

  def get_init_state(self):
    return self.h0

  def step(self, x_t, h_prev):
    input_t = self.f_embedding(x_t)
    z_t = T.nnet.sigmoid(T.dot(input_t, self.wz) + T.dot(h_prev, self.uz))
    r_t = T.nnet.sigmoid(T.dot(input_t, self.wr) + T.dot(h_prev, self.ur))
    h_tilde_t = T.nnet.sigmoid(T.dot(input_t, self.w) + r_t * T.dot(h_prev, self.u))
    h_t = z_t * h_prev + (1 - z_t) * h_tilde_t
    return h_t

  def write(self, h_t):
    return T.nnet.softmax(T.dot(h_t, self.w_out))[0]
