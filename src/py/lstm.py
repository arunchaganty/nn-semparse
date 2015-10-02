"""IRW model powered by an LSTM."""
import numpy
import random
import sys
import theano
from theano.ifelse import ifelse
from theano import tensor as T
from spec import IRWSpec

class LSTMIRWSpec(IRWSpec):
  """RNN with LSTM and IRW.

  Parameters are named after functions in the irw-doc.pdf file.
  """
  def create_vars(self):
    # Initial state
    # The hidden state must store both c_t, the memory cell, 
    # and h_t, what we normally call the hidden state
    self.h0 = theano.shared(
        name='h0', 
        value=0.2 * numpy.random.uniform(-1.0, 1.0, 2 * self.nh).astype(theano.config.floatX))

    # Encoder hidden state updates
    self.wi_enc = theano.shared(
        name='wi_enc',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.de_total, self.nh)).astype(theano.config.floatX))
    self.ui_enc = theano.shared(
        name='ui_enc',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
    self.wf_enc = theano.shared(
        name='wf_enc',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.de_total, self.nh)).astype(theano.config.floatX))
    self.uf_enc = theano.shared(
        name='uf_enc',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
    self.wo_enc = theano.shared(
        name='wo_enc',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.de_total, self.nh)).astype(theano.config.floatX))
    self.uo_enc = theano.shared(
        name='uo_enc',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
    self.wc_enc = theano.shared(
        name='wc_enc',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.de_total, self.nh)).astype(theano.config.floatX))
    self.uc_enc = theano.shared(
        name='uc_enc',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))

    # Decoder hidden state updates
    self.wi_dec = theano.shared(
        name='wi_dec',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.de_total, self.nh)).astype(theano.config.floatX))
    self.ui_dec = theano.shared(
        name='ui_dec',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
    self.wf_dec = theano.shared(
        name='wf_dec',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.de_total, self.nh)).astype(theano.config.floatX))
    self.uf_dec = theano.shared(
        name='uf_dec',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
    self.wo_dec = theano.shared(
        name='wo_dec',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.de_total, self.nh)).astype(theano.config.floatX))
    self.uo_dec = theano.shared(
        name='uo_dec',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
    self.wc_dec = theano.shared(
        name='wc_dec',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.de_total, self.nh)).astype(theano.config.floatX))
    self.uc_dec = theano.shared(
        name='uc_dec',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))

    # Logistic regression node to choose read or write
    self.w_b = theano.shared(
        name='w_b', 
        value=0.2 * numpy.random.uniform(-1.0, 1.0, 2 * self.nh).astype(theano.config.floatX))

    # Softmax layer to choose what to write
    self.w_c = theano.shared(
        name='w_c', 
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (2 * self.nh, self.nw_out)).astype(theano.config.floatX))

    # Params
    self.params = [
        self.h0,
        self.wi_enc, self.ui_enc, self.wf_enc, self.uf_enc,
        self.wo_enc, self.uo_enc, self.wc_enc, self.uc_enc,
        self.wi_dec, self.ui_dec, self.wf_dec, self.uf_dec,
        self.wo_dec, self.uo_dec, self.wc_dec, self.uc_dec,
        self.w_b,
        self.w_c,
    ]

  def unpack(self, hidden_state):
    c_t = hidden_state[0:self.nh]
    h_t = hidden_state[self.nh:]
    return (c_t, h_t)

  def pack(self, c_t, h_t):
    return T.concatenate([c_t, h_t])

  def _lstm_fn(self, input_t, c_h_prev, wi=None, ui=None, wf=None, uf=None,
               wo=None, uo=None, wc=None, uc=None):
    c_prev, h_prev = self.unpack(c_h_prev)
    i_t = T.nnet.sigmoid(T.dot(input_t, wi) + T.dot(h_prev, ui))
    f_t = T.nnet.sigmoid(T.dot(input_t, wf) + T.dot(h_prev, uf))
    o_t = T.nnet.sigmoid(T.dot(input_t, wo) + T.dot(h_prev, uo))
    c_tilde_t = T.tanh(T.dot(input_t, wc) + T.dot(h_prev, uc))

    c_t = f_t * c_prev + i_t * c_tilde_t
    h_t = o_t * T.tanh(c_t)
    return self.pack(c_t, h_t)

  def f_rnn(self, r_t, w_t, c_h_prev):
    r_emb = self.f_read_embedding(r_t)
    w_emb = self.f_write_embedding(w_t)
    input_t = T.concatenate([r_emb, w_emb])
    h_t_encode = self._lstm_fn(
        input_t, c_h_prev, wi=self.wi_enc, ui=self.ui_enc,
        wf=self.wf_enc, uf=self.uf_enc, wo=self.wo_enc, uo=self.uo_enc,
        wc=self.wc_enc, uc=self.uc_enc)
    h_t_decode = self._lstm_fn(
        input_t, c_h_prev, wi=self.wi_dec, ui=self.ui_dec,
        wf=self.wf_dec, uf=self.uf_dec, wo=self.wo_dec, uo=self.uo_dec,
        wc=self.wc_dec, uc=self.uc_dec)
    # Use encoder if we did a read, otherwise use the decoder.
    h_t = ifelse(T.ge(r_t, 0), h_t_encode, h_t_decode)
    return h_t

  def f_p_read(self, c_h):
    return T.nnet.sigmoid(T.dot(c_h, self.w_b))

  def f_dist_write(self, c_h):
    return T.nnet.softmax(T.dot(c_h, self.w_c))[0]
