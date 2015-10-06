"""Specifies a particular instance of a model."""
import numpy
import pickle

class Spec(object):
  """Abstract class for a specification of a sequence-to-sequence RNN model.

  Concrete sublcasses must implement the following methods:
  - self.create_vars(): called by __init__, should initialize parameters.
  - self.get_local_params(): Get all local parameters (excludes vocabulary).
  """
  def __init__(self, in_vocabulary, out_vocabulary, hidden_size):
    """Initialize.

    Args:
      in_vocabulary: list of words in the vocabulary of the input
      out_vocabulary: list of words in the vocabulary of the output
      embedding_dim: dimension of word vectors
      hidden_size: dimension of hidden layer
    """
    self.in_vocabulary = in_vocabulary
    self.out_vocabulary = out_vocabulary
    self.hidden_size = hidden_size
    self.create_vars()

  def set_in_vocabulary(self, in_vocabulary):
    # TODO: make this work
    raise NotImplementedError
    #self.in_vocabulary = in_vocabulary
    #self.de_in = embedding_dim = in_vocabulary.emb_size
    #self.nw_in = in_vocabulary.size()

  def set_out_vocabulary(self, out_vocabulary):
    # TODO: make this work
    raise NotImplementedError
    #self.out_vocabulary = out_vocabulary
    #self.de_out = embeddoutg_dim = out_vocabulary.emb_size
    #self.nw_out = out_vocabulary.size()

  def create_vars(self):
    raise NotImplementedError

  def get_local_params(self):
    raise NotImplementedError

  def f_read_embedding(self, i):
    return self.in_vocabulary.get_theano_embedding(i)

  def f_write_embedding(self, i):
    return self.out_vocabulary.get_theano_embedding(i)

  def get_params(self):
    """Get all parameters (things we optimize with respect to)."""
    return (self.get_local_params()
            + self.in_vocabulary.get_theano_params()
            + self.out_vocabulary.get_theano_params())

  def get_all_shared(self):
    """Get all shared theano varaibles.

    There are shared variables that we do not necessarily optimize respect to,
    but may be held fixed (e.g. GloVe vectors, if desired).
    We don't backpropagate through these, but we do need to feed them to scan.
    """
    return (self.get_local_params() 
            + self.in_vocabulary.get_theano_all()
            + self.out_vocabulary.get_theano_all())

  def save(self, filename):
    """Save the parameters to a filename."""
    with open(filename, 'w') as f:
      pickle.dump(self, f)

def load(filename):
  with open(filename) as f:
    return pickle.load(f)
