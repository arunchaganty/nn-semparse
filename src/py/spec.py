"""Specifies the particular flavor of IRW RNN."""
import numpy
import pickle

class IRWSpec(object):
  """Abstract class for a specification of an IRW RNN.

  Concrete sublcasses must implement the following methods:
  - self.create_vars(): first thing done by __init__, 
        should set self.params to be a list of parameters to optimize by SGD.
  - self.f_rnn(r_t, w_t, h_prev): returns new hidden state
  - self.f_p_read(h): returns boolean probability to execute a read action.
  - self.f_dist_write(h): returns distribution over words to write.
  """
  def __init__(self, in_vocabulary, out_vocabulary, hidden_size):
    """Initialize.

    Args:
      in_vocabulary: list of words in the vocabulary of the input
      out_vocabulary: list of words in the vocabulary of the output
      embedding_dim: dimension of word vectors
      hidden_size: dimension of hidden layer

    Convention used by this class:
      nh: dimension of hidden layer
      nw: number of words in the vocabulary
      de: dimension of word embeddings
    """
    self.in_vocabulary = in_vocabulary
    self.de_in = embedding_dim = in_vocabulary.emb_size
    self.nw_in = in_vocabulary.size()

    self.out_vocabulary = out_vocabulary
    self.de_out = embedding_dim = out_vocabulary.emb_size
    self.nw_out = out_vocabulary.size()

    self.de_total = self.de_in + self.de_out
    self.nh = hidden_size
    self.create_vars()

  def set_in_vocabulary(self, in_vocabulary):
    self.in_vocabulary = in_vocabulary
    self.de_in = embedding_dim = in_vocabulary.emb_size
    self.nw_in = in_vocabulary.size()

  def set_out_vocabulary(self, out_vocabulary):
    self.out_vocabulary = out_vocabulary
    self.de_out = embeddoutg_dim = out_vocabulary.emb_size
    self.nw_out = out_vocabulary.size()

  def create_vars(self):
    raise NotImplementedError

  def f_rnn(self, r_t, w_t, h_prev):
    raise NotImplementedError

  def f_p_read(self, h):
    raise NotImplementedError

  def f_dist_write(self, h):
    raise NotImplementedError

  def f_read_embedding(self, i):
    return self.in_vocabulary.get_theano_embedding(i)

  def f_write_embedding(self, i):
    return self.out_vocabulary.get_theano_embedding(i)

  def get_params(self):
    return (self.in_vocabulary.get_theano_params()
            + self.out_vocabulary.get_theano_params()
            + self.params)

  def get_regularization(self, lambda_reg):
    # By default, no regularization
    return None


  def get_all_shared(self):
    # There are shared variables that we do not optimize respect to,
    # but are instead held fixed (e.g. GloVe vectors, if desired).
    # They will not be backpropagated through, but we do need to feed them to scan.
    return (self.params 
            + self.in_vocabulary.get_theano_all()
            + self.out_vocabulary.get_theano_all())

  def save(self, filename):
    """Save the parameters to a filename."""
    with open(filename, 'w') as f:
      pickle.dump(self, f)

  @classmethod
  def load(cls, filename):
    with open(filename) as f:
      return pickle.load(f)
