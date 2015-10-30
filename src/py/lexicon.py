"""A lexicon maps input substrings to an output token."""
import collections
import numpy
import theano

class Lexicon(object):
  """A lexicon object.

  For now, maintains its own embeddings, just like Vocabulary.
  TODO(robinjia): This will probably change later, as we want
  to generate the embeddings based on local context.

  TODO(robinjia): enable mapping to a sequence of output tokens.
  """
  def __init__(self, entries, emb_size):
    """Create the lexicon.

    entries should be a list of (input, output) pairs.
    """
    self.entries = entries
    self._map = collections.defaultdict(list)
    for e in entries:
      self._map[e[0]].append(e[1])

    # Embedding matrix
    self.emb_mat = theano.shared(
        name='lexicon_emb_mat',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.size(), emb_size)).astype(theano.config.floatX))

  def get_outputs(self, in_str):
    if in_str not in self._map: return []
    return self._map[in_str]

  def size(self):
    return len(self.entries)

  def get_theano_embedding(self, i):
    return self.emb_mat[i]

  def get_theano_params(self):
    return [self.emb_mat]

  def get_theano_all(self):
    return [self.emb_mat]

  def get_word(self, i):
    """Get the output word associated with this lexicon entry."""
    return self.entries[i][1]

  @classmethod
  def from_vocabulary(cls, vocab, emb_size):
    """Create a lexicon that just maps every word in a vocabulary to itself."""
    entries = [(w, w) for w in vocab.word_list]
    lex = cls(entries, emb_size)
    print 'Extract lexicon of size %d' % lex.size()
    return lex
