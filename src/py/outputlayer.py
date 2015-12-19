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
    nl: number of lexicon entries to create embeddings for.
  """ 
  def __init__(self, vocab, lexicon, hidden_size):
    self.vocab = vocab
    self.lexicon = lexicon
    self.de = vocab.emb_size
    self.nh = hidden_size
    self.nw = vocab.size()
    if lexicon:
      self.nl = lexicon.num_embeddings
    else:
      self.nl = 0
    self.create_vars()

  def create_vars(self):
    self.w_out = theano.shared(
        name='w_out', 
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.nw, self.nh)).astype(theano.config.floatX))
        # Each row is one word
    self.params = [self.w_out]

    if self.lexicon:
      self.w_lex = theano.shared(
          name='w_lex', 
          value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.nl, self.nh)).astype(theano.config.floatX))
      self.params.append(self.w_lex)

  def write(self, h_t, cur_lex_entries, attn_scores=None):
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
      attn_scores: unnormalized scores from the attention module, if doing 
          attention-based copying.
    """
    def f(i, *params):
      # Concatenate embeddings of all lexicon entries in cur_lex_entries
      return self.w_lex[i]
    if self.lexicon:
      lex_embs, _ = theano.scan(f, sequences=[cur_lex_entries],
                                non_sequences=self.w_lex)
      big_mat = T.concatenate([self.w_out, lex_embs])  # total_num_words x self.nh
      mat = ifelse(T.eq(cur_lex_entries.shape[0], 0), self.w_out, big_mat)
      return T.nnet.softmax(T.dot(h_t, mat.T))[0]
    elif attn_scores:
      scores = T.dot(h_t, self.w_out.T)
      return T.nnet.softmax(T.concatenate([scores, attn_scores]))[0]
    else:
      return T.nnet.softmax(T.dot(h_t, self.w_out.mat.T))[0]
