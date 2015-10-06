"""A vanilla RNN layer."""
import numpy
import random
import sys
import theano
from theano.ifelse import ifelse
from theano import tensor as T

from rnnlayer import RNNLayer

class VanillaRNNLayer(RNNLayer):
  """A standard vanilla RNN layer."""
  def create_vars(self, create_init_state, create_output_layer):
    # Initial state
    if create_init_state:
      self.h0 = theano.shared(
          name='h0', 
          value=0.2 * numpy.random.uniform(-1.0, 1.0, self.nh).astype(theano.config.floatX))
      init_state_params = [self.h0]
    else:
      init_state_params = []

    # Recurrent layer
    self.u_x = theano.shared(
        name='u_x',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.de_in, self.nh)).astype(theano.config.floatX))
    self.u_h = theano.shared(
        name='u_h',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
    recurrence_params = [self.u_x, self.u_h]

    # Output layer
    if create_output_layer:
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
    h_t = T.nnet.sigmoid(T.dot(h_prev, self.u_h) + T.dot(input_t, self.u_x))
    return h_t

  def write(self, h_t):
    return T.nnet.softmax(T.dot(h_t, self.w_out))[0]
