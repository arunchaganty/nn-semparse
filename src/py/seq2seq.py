"""Generic class for sequence-to-sequence neural net model."""
import collections
import itertools
import math
import numpy
import random
import sys
import theano
from theano import tensor as T

import sdf
from vocabulary import Vocabulary

theano.config.mode='FAST_RUN'
#theano.config.mode='FAST_COMPILE'
random.seed(0)


TestExample = collections.namedtuple('TestExample', ['utterance', 'candidates'])

class SequenceToSequenceModel(NeuralModel):
  """Abstract class for sequence-to-sequence neural network.

  Implementing classes must implement the following functions:
  - self.create_vars(): first thing done by __init__, 
        should set self.params to be a list of parameters to optimize by SGD.
  - self.encode_h(x_t, h_prev): returns h_t
  - self.decode_s(h): returns s, the probability distribution over words
  - self.decode_h(y_t, h_prev): returns h_t
  """
  def __init__(self, vocabulary, embedding_dim, hidden_size, reverse_source=True):
    """Initialize.

    Args:
      vocabulary: list of words in the vocabulary
      embedding_dim: dimension of word vectors
      hidden_size: dimension of hidden layer
      reverse_source: whether to reverse the source sentence

    Convention used by this class:
      nh: dimension of hidden layer
      nw: number of words in the vocabulary
      de: dimension of word embeddings
    """
    self.vocabulary = vocabulary
    self.de = embedding_dim 
    self.nw = vocabulary.size()
    self.nh = hidden_size
    self.reverse_source = reverse_source

    self.create_vars()
    self.setup()

  def create_vars(self):
    raise NotImplementedError

  def encode_h(self, x_t, h_prev):
    raise NotImplementedError

  def decode_s(self, h):
    raise NotImplementedError

  def decode_h(self, y_t, h_prev):
    raise NotImplementedError

  def train_one(self, example, eta):
    x_inds, y_inds = example
    nll = self._train(x_inds, y_inds, eta)
    return nll

  def get_score(self, x_inds, y_inds):
    # Convert negative loglikelihood to probability
    return math.exp(-self._get_nll(x_inds, y_inds))

  def setup(self):
    """Set up the model."""
    # Encoding
    def recurrence_enc(x_t, h_prev):
      return self.encode_h(x_t, h_prev)
    x_inds = T.lvector()
    x = self.emb_mat[x_inds].reshape((x_inds.shape[0], self.de))
    h_enc, _ = theano.scan(fn=recurrence_enc, sequences=x,
                           outputs_info=self.h0, n_steps=x.shape[0])
    h_T = h_enc[-1]
    self._encode = theano.function(inputs=[x_inds], outputs=h_T)

    # Decoding
    def recurrence_dec(y_t, h_prev):
      s_prev = self.decode_s(h_prev)
      h_t = self.decode_h(y_t, h_prev)
      return [h_t, s_prev]
    y_inds = T.lvector()
    y = self.emb_mat[y_inds].reshape((y_inds.shape[0], self.de))
    [h_dec, s], _ = theano.scan(fn=recurrence_dec, sequences=y,
                                outputs_info=[h_T, None], n_steps=y.shape[0])
    p_y_given_x_sentence = s[:, 0, :]

    # Compute likelihood and gradient for training
    # Report the sum of log likelihoods, but for gradient use the mean,
    # so that the step size is roughly the same independent of sentence length
    # (otherwise norm of gradient scales linearly with sentence length)
    neg_loglikelihood_mean = -T.mean(T.log(p_y_given_x_sentence)[T.arange(y.shape[0]), y_inds])
    neg_loglikelihood_sum = -T.sum(T.log(p_y_given_x_sentence)[T.arange(y.shape[0]), y_inds])
    gradients = T.grad(neg_loglikelihood_mean, self.params)
    lr = T.scalar('lr')
    updates = collections.OrderedDict(
        (p, p - lr * T.clip(g, -1, 1)) for p, g in itertools.izip(self.params, gradients))
    self._get_nll = theano.function(inputs=[x_inds, y_inds], outputs=neg_loglikelihood_sum)
    self._train = theano.function(inputs=[x_inds, y_inds, lr], outputs=neg_loglikelihood_sum,
                                  updates=updates)

    # Predict output given input
    h_cur = T.dvector()
    y_next = T.lscalar()
    y_next_vec = self.emb_mat[y_next].reshape((1, self.de))
    s_cur = self.decode_s(h_cur)
    h_next = self.decode_h(y_next_vec, h_cur)
    self._decode_one = theano.function(name='_decode_one', inputs=[h_cur], outputs=s_cur)
    self._decoder_advance = theano.function(name='_decoder_advance',
                                            inputs=[h_cur, y_next], outputs=h_next)

  def decode(self, x_inds, max_len=100, stop=0):
    h_cur = self._encode(x_inds)
    y_inds = []
    prob = 1
    for i in range(max_len):
      s_cur = self._decode_one(h_cur)[0,:]
      y_cur = max(enumerate(s_cur), key=lambda x: x[1])[0]
      y_inds.append(y_cur)
      prob *= s_cur[y_cur]
      if y_cur == stop: break
      h_cur = self._decoder_advance(h_cur, y_cur).flatten()
    return y_inds

  def sdf_to_train_data(self, sdf_data):
    """Convert SDF dataset into one fit for training the model.

    Train dataset format is list of examples, where each example
    is a list of indices in the vocabulary.
    """
    train_data = []
    for records in sdf_data:
      utterance = records[0].utterance
      best_candidate = sdf.get_best_correct(records)
      if best_candidate:
        canonical_utterance = best_candidate.canonical_utterance
        x_inds = self.vocabulary.sentence_to_indices(utterance)
        if self.reverse_source:
          x_inds = x_inds[::-1]
        y_inds = self.vocabulary.sentence_to_indices(canonical_utterance)
        train_data.append((x_inds, y_inds))
    return train_data
