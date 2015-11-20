"""Specifies a particular instance of a soft attention model.

We use the global attention model with input feeding
used by Luong et al. (2015).
See http://stanford.edu/~lmthang/data/papers/emnlp15_attn.pdf
"""
import numpy
import theano
from theano import tensor as T

from outputlayer import OutputLayer
from spec import Spec

class AttentionSpec(Spec):
  """Abstract class for a specification of an encoder-decoder model.
  
  Concrete subclasses must implement the following method:
  - self.create_rnn_layer(vocab, hidden_size): Create an RNN layer.
  """
  def create_vars(self):
    # TODO: move this into lstm.py
    if self.rnn_type == 'lstm':
      annotation_size = 4 * self.hidden_size
      dec_full_size = 2 * self.hidden_size
    else:
      annotation_size = 2 * self.hidden_size
      dec_full_size = self.hidden_size

    self.fwd_encoder = self.create_rnn_layer(
        self.hidden_size, self.in_vocabulary.emb_size,
        self.in_vocabulary.size(), True)
    self.bwd_encoder = self.create_rnn_layer(
        self.hidden_size, self.in_vocabulary.emb_size,
        self.in_vocabulary.size(), True)
    self.decoder = self.create_rnn_layer(
        self.hidden_size, self.out_vocabulary.emb_size + annotation_size,
        self.out_vocabulary.size(), False)
    self.writer = self.create_output_layer(self.out_vocabulary, self.lexicon,
                                           self.hidden_size + annotation_size)

    self.w_enc_to_dec = theano.shared(
        name='w_enc_to_dec',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (dec_full_size, annotation_size)).astype(theano.config.floatX))
    self.w_attention = theano.shared(
        name='w_attention',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.hidden_size, annotation_size)).astype(theano.config.floatX))

  def get_local_params(self):
    return (self.fwd_encoder.params + self.bwd_encoder.params + 
            self.decoder.params + self.writer.params + [self.w_enc_to_dec])

  def create_output_layer(self, vocab, lexicon, hidden_size):
    return OutputLayer(vocab, lexicon, hidden_size)

  def get_init_fwd_state(self):
    return self.fwd_encoder.get_init_state()

  def get_init_bwd_state(self):
    return self.bwd_encoder.get_init_state()

  def f_enc_fwd(self, x_t, h_prev):
    """Returns the next hidden state for forward encoder."""
    input_t = self.in_vocabulary.get_theano_embedding(x_t)
    return self.fwd_encoder.step(input_t, h_prev)

  def f_enc_bwd(self, x_t, h_prev):
    """Returns the next hidden state for backward encoder."""
    input_t = self.in_vocabulary.get_theano_embedding(x_t)
    return self.bwd_encoder.step(input_t, h_prev)

  def get_dec_init_state(self, enc_last_state):
    return T.dot(self.w_enc_to_dec, enc_last_state)
    #return T.tanh(T.dot(self.w_enc_to_dec, enc_last_state))

  def f_dec(self, y_t, c_prev, h_prev):
    """Returns the next hidden state for decoder."""
    y_emb_t = self.out_vocabulary.get_theano_embedding(y_t)
    input_t = T.concatenate([y_emb_t, c_prev])
    return self.decoder.step(input_t, h_prev)

  def get_alpha(self, h_for_write, annotations):
    scores = T.dot(T.dot(self.w_attention, annotations.T).T, h_for_write)
    alpha = T.nnet.softmax(scores)[0]
    return alpha

  def get_context(self, alpha, annotations):
    c_t = T.dot(alpha, annotations)
    return c_t

  def f_write(self, h_t, c_t, cur_lex_entries):
    """Gives the softmax output distribution."""
    input_t = T.concatenate([h_t, c_t])
    return self.writer.write(input_t, cur_lex_entries)
