"""An output layer."""
import numpy
import theano
from theano.ifelse import ifelse
import theano.tensor as T

class OutputLayer(object):
  """Class that sepcifies parameters of an output layer.
  
  Conventions used by this class (shared with spec.py):
    nh: dimension of hidden layer
    nw: number of words in the vocabulary
    de: dimension of word embeddings
  """ 
  def __init__(self, vocab, lexicon, hidden_size):
    self.vocab = vocab
    self.lexicon = lexicon
    self.de = vocab.emb_size
    self.nh = hidden_size
    self.nw = vocab.size()
    self.create_vars()

  def create_vars(self):
    self.w_out = theano.shared(
        name='w_out', 
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.nw, self.nh)).astype(theano.config.floatX))
        # Each row is one word
    self.params = [self.w_out]

  def write(self, h_t, cur_lex_entries):
    """Get a distribution over words to write.
    
    We allow the output layer to write an output word or use a lexicon entry.
    Therefore, this actually returns a vector of length 
      
        nw + len(cur_lex_entries).

    Entries in [0, nw) are probablity of emitting i-th output word,
    and entries in [nw, nw + len(cur_lex_entries))
    are probability of using (i - nw)-th entry in cur_lex_entries.

    Currently, we just have a single representation for each lexicon entry.
    TODO(robinjia): make lexicon entry representations depend on input sentence.

    Args:
      h_t: theano vector representing hidden state
      cur_lex_entries: theano lvector that lists which lexicon entries are active
        for the current example.
        Each element of cur_lex_entries should be in [0, lexicon.size())
    """
    # Concatenate embeddings of all lexicon entries in cur_lex_entries
    def f(i, *params):
      return self.lexicon.get_theano_embedding(i)
    if self.lexicon:
      cur_embs, _ = theano.scan(f, sequences=[cur_lex_entries],
                                non_sequences=self.lexicon.get_theano_params())
      big_mat = T.concatenate([self.w_out, cur_embs])  # total_num_words x self.nh
      mat = ifelse(T.eq(cur_lex_entries.shape[0], 0), self.w_out, big_mat)
    else:
      mat = self.w_out
    return T.nnet.softmax(T.dot(h_t, mat.T))[0]
