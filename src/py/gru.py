"""IRW model powered by gated recursive unit."""
import numpy
import random
import sys
import theano
from theano.ifelse import ifelse
from theano import tensor as T
from spec import IRWSpec

class GRUIRWSpec(IRWSpec):
  """RNN with gated recursive unit and IRW.

  Parameters are named after functions in the irw-doc.pdf file.
  """
  def create_vars(self):
    # Initial state
    self.h0 = theano.shared(
        name='h0', 
        value=0.2 * numpy.random.uniform(-1.0, 1.0, self.nh).astype(theano.config.floatX))

    # Encoder hidden state updates
    self.wz_enc = theano.shared(
        name='wz_enc',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.de_total, self.nh)).astype(theano.config.floatX))
    self.uz_enc = theano.shared(
        name='uz_enc',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
    self.wr_enc = theano.shared(
        name='wr_enc',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.de_total, self.nh)).astype(theano.config.floatX))
    self.ur_enc = theano.shared(
        name='ur_enc',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
    self.w_enc = theano.shared(
        name='w_enc',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.de_total, self.nh)).astype(theano.config.floatX))
    self.u_enc = theano.shared(
        name='u_enc',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))

    # Decoder hidden state updates
    self.wz_dec = theano.shared(
        name='wz_dec',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.de_total, self.nh)).astype(theano.config.floatX))
    self.uz_dec = theano.shared(
        name='uz_dec',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
    self.wr_dec = theano.shared(
        name='wr_dec',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.de_total, self.nh)).astype(theano.config.floatX))
    self.ur_dec = theano.shared(
        name='ur_dec',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
    self.w_dec = theano.shared(
        name='w_dec',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.de_total, self.nh)).astype(theano.config.floatX))
    self.u_dec = theano.shared(
        name='u_dec',
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
        self.wz_enc, self.uz_enc, self.wr_enc, self.ur_enc, self.w_enc, self.u_enc,
        self.wz_dec, self.uz_dec, self.wr_dec, self.ur_dec, self.w_dec, self.u_dec,
        self.w_b,
        self.w_c,
    ]

  def _gru_fn(self, input_t, h_prev,
              wz=None, uz=None, wr=None, ur=None, w=None, u=None):
    z_t = T.nnet.sigmoid(T.dot(input_t, wz) + T.dot(h_prev, uz))
    r_t = T.nnet.sigmoid(T.dot(input_t, wr) + T.dot(h_prev, ur))
    h_tilde_t = T.nnet.sigmoid(T.dot(input_t, w) + r_t * T.dot(h_prev, u))
    h_t = z_t * h_prev + (1 - z_t) * h_tilde_t
    return h_t


  def f_rnn(self, r_t, w_t, h_prev):
    r_emb = self.f_read_embedding(r_t)
    w_emb = self.f_write_embedding(w_t)
    input_t = T.concatenate([r_emb, w_emb])
    h_t_encode = self._gru_fn(input_t, h_prev, wz=self.wz_enc, uz=self.uz_enc,
                              wr=self.wr_enc, ur=self.ur_enc,
                              w=self.w_enc, u=self.u_enc)
    h_t_decode = self._gru_fn(input_t, h_prev, wz=self.wz_dec, uz=self.uz_dec,
                              wr=self.wr_dec, ur=self.ur_dec,
                              w=self.w_dec, u=self.u_dec)
    # Use encoder if we did a read, otherwise use the decoder.
    h_t = ifelse(T.ge(r_t, 0), h_t_encode, h_t_decode)
    return h_t

  def f_p_read(self, h):
    return T.nnet.sigmoid(T.dot(h, self.w_b))

  def f_dist_write(self, h):
    return T.nnet.softmax(T.dot(h, self.w_c))[0]

  def get_regularization(self, lambda_reg):
    # Apply L2 to the output layers, to prevent NaNs.
    return lambda_reg * (T.sum(self.w_b ** 2) + T.sum(self.w_c ** 2))
