"""Abstract class that specifies parameters of a recurrent layer."""

class RNNLayer(object):
  """Abstract class that sepcifies parameters of a recurrent layer.
  
  Conventions used by this class (shared with spec.py):
    nh: dimension of hidden layer
    nw: number of words in the vocabulary
    de: dimension of word embeddings
  """ 
  def __init__(self, vocab, hidden_size, 
               create_init_state=False):
    self.vocab = vocab
    self.de = vocab.emb_size
    self.nh = hidden_size
    self.nw = vocab.size()
    self.create_vars(create_init_state)

  def create_vars(self):
    raise NotImplementedError

  def get_init_state(self):
    raise NotImplementedError

  def step(self, x_t, h_prev):
    raise NotImplementedError

  def get_h_for_write(self, h):
    """Override if only want to expose part of hidden state for output."""
    return h

  def f_embedding(self, i):
    return self.vocab.get_theano_embedding(i)
