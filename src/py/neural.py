"""A generic continuous neural sequence-to-sequence model."""
import collections
import itertools
import numpy
import os
import random
import sys
import theano
from theano.ifelse import ifelse
from theano import tensor as T
import time

CLIP_THRESH = 3.0  # Clip gradient if norm is larger than this

class NeuralModel(object):
  """A generic continuous neural sequence-to-sequence model.

  Implementing classes must implement the following functions:
    - self.setup(): set up the model.
    - self.get_objective_and_gradients(x, y): Get objective and gradients.
    - self.decode_greedy(ex, max_len=100): Do a greedy decoding of x, predict y.
    - self.decode_greedy(ex, beam_size=1, max_len=100): Beam search to predict y

  They should also override
    - cls.get_spec_class(): The associated Spec subclass for this NeuralModel.

  Convention used by this class:
    nh: dimension of hidden layer
    nw: number of words in the vocabulary
    de: dimension of word embeddings
  """
  def __init__(self, spec, float_type=numpy.float64):
    """Initialize.

    Args:
      spec: Spec object.
      float_type: Floating point type (default 64-bit/double precision)
    """
    self.spec = spec
    self.in_vocabulary = spec.in_vocabulary
    self.out_vocabulary = spec.out_vocabulary
    self.lexicon = spec.lexicon
    self.float_type = float_type
    self.params = spec.get_params()
    if spec.step_rule in ('adagrad', 'rmsprop'):
      # Initialize the grad norm cache
      self.grad_norm_cache = [
          theano.shared(
              name='%s_norm_cache' % p.name,
              value=numpy.zeros_like(p.get_value()))
          for p in self.params]
    self.all_shared = spec.get_all_shared()

    self.setup()
    print >> sys.stderr, 'Setup complete.'

  @classmethod
  def get_spec_class(cls):
    raise NotImplementedError

  def setup(self):
    """Do all necessary setup (e.g. compile theano functions)."""
    raise NotImplementedError

  def sgd_step(self, ex, eta):
    """Perform an SGD step.

    This is a default implementation, which assumes
    a function called self._backprop() which updates parameters.

    Override if you need a more complicated way of computing the 
    objective or updating parameters.

    Returns: the current objective value
    """
    print 'x: %s' % ex.x_str
    print 'y: %s' % ex.y_str
    info = self._backprop(ex.x_inds, ex.y_inds, eta, ex.lex_inds, 
                          ex.y_lex_inds, ex.y_in_x_inds)
    p_y_seq = info[0]
    log_p_y = info[1]
    objective = -log_p_y
    print 'P(y_i): %s' % p_y_seq
    return objective

  def decode_greedy(self, ex, max_len=100):
    """Decode input greedily.
    
    Returns list of (prob, y_tok_seq) pairs."""
    raise NotImplementedError

  def decode_beam(self, ex, beam_size=1, max_len=100):
    """Decode input with beam search.
    
    Returns list of (prob, y_tok_seq) pairs."""
    raise NotImplementedError

  def on_train_epoch(self, t):
    """Optional method to do things every epoch."""
    for p in self.params:
      print '%s: %s' % (p.name, p.get_value())

  def train(self, dataset, eta=0.1, T=[], verbose=False):
    # train with SGD (batch size = 1)
    cur_lr = eta
    max_iters = sum(T)
    lr_changes = set([sum(T[:i]) for i in range(1, len(T))])
    for it in range(max_iters):
      t0 = time.time()
      if it in lr_changes:
        # Halve the learning rate
        cur_lr = 0.5 * cur_lr
      total_nll = 0
      random.shuffle(dataset)
      for ex in dataset:
        nll = self.sgd_step(ex, eta)
        total_nll += nll
      self.on_train_epoch(it)
      t1 = time.time()
      print 'NeuralModel.train(): iter %d (lr = %g): total nll = %g (%g seconds)' % (
          it, cur_lr, total_nll, t1 - t0)
