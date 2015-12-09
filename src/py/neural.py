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
    self.all_shared = spec.get_all_shared()

    self.setup()
    print >> sys.stderr, 'Setup complete.'

  @classmethod
  def get_spec_class(cls):
    raise NotImplementedError

  def setup(self):
    """Do all necessary setup (e.g. compile theano functions)."""
    raise NotImplementedError

  def get_objective_and_gradients(self, ex):
    """Get objective and gradients.

    This is a default implementation, which assumes
    a function called self._backprop().

    Override if you need a more complicated way of computing the 
    objective and gradient.

    Returns: tuple (objective, gradients) where
      objective: the current objective value
      gradients: map from parameter to gradient
    """
    info = self._backprop(ex.x_inds, ex.y_inds, ex.lex_inds, ex.y_lex_inds, ex.y_in_x_inds)
    p_y_seq = info[0]
    log_p_y = info[1]
    gradients_list = info[2:]
    objective = -log_p_y
    gradients = dict(itertools.izip(self.params, gradients_list))
    print 'P(y_i): %s' % p_y_seq
    return (objective, gradients)

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

  def train(self, dataset, eta=0.1, T=10, verbose=False, batch_size=1):
    # batch_size = size for mini batch.  Defaults to SGD.
    for it in range(T):
      t0 = time.time()
      total_nll = 0
      random.shuffle(dataset)
      for i in range(0, len(dataset), batch_size):
        do_update = i + batch_size <= len(dataset)
        cur_examples = dataset[i:(i+batch_size)]
        nll = self._train_batch(cur_examples, eta, do_update=do_update)
        total_nll += nll
        if verbose:
          print 'NeuralModel.train(): iter %d, example = %s: nll = %g' % (
              it, str(ex), nll)
      t1 = time.time()
      print 'NeuralModel.train(): iter %d: total nll = %g (%g seconds)' % (
          it, total_nll, t1 - t0)
      self.on_train_epoch(it)

  def _train_batch(self, examples, eta, do_update=True):
    """Run training given a batch of training examples.
    
    Returns negative log-likelihood.
    If do_update is False, compute nll but don't do the gradient step.
    """
    objective = 0
    gradients = {}
    for ex in examples:
      print 'x: %s' % ex.x_str
      print 'y: %s' % ex.y_str
      cur_objective, cur_gradients = self.get_objective_and_gradients(ex)
      objective += cur_objective
      for p in self.params:
        if p in gradients:
          gradients[p] += cur_gradients[p] / len(examples)
        else:
          gradients[p] = cur_gradients[p] / len(examples)
    if do_update:
      for p in self.params:
        self._perform_sgd_step(p, gradients[p], eta)
    return objective

  def _perform_sgd_step(self, param, gradient, eta):
    """Do a gradient descent step."""
    #print param.name
    #print param.get_value()
    #print gradient
    old_value = param.get_value()
    grad_norm = numpy.sqrt(numpy.sum(gradient**2))
    if grad_norm >= CLIP_THRESH:
      gradient = gradient * CLIP_THRESH / grad_norm
      new_norm = numpy.sqrt(numpy.sum(gradient**2))
      print 'Clipped norm of %s from %g to %g' % (param, grad_norm, new_norm)
    new_value = old_value - eta * gradient
    param.set_value(new_value)

