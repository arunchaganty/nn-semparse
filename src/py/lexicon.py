"""A lexicon maps input substrings to an output token."""
import collections
import numpy
import theano

class Lexicon(object):
  """A lexicon object.

  For now, maintains its own embeddings, just like Vocabulary.
  TODO(robinjia): This will probably change later, as we want
  to generate the embeddings based on local context.

  In this class, an "entry" is a (input, output) 2-tuple.

  TODO(robinjia): enable mapping to a sequence of output tokens.
  """
  def __init__(self, entries, emb_size, unk_func=None):
    """Create the lexicon.

    Args:
      entries: list of (input, output) pairs.
      emb_size: size of embeddings
      unk_func: If provided, this function says whether to map entry to UNK embedding.
    """
    self.entries = entries
    self.entry_to_index = {}
    self.entry_map = collections.defaultdict(list)  # Map input to entry
    cur_ind = 1
    for e in entries:
      self.entry_map[e[0]].append(e)
      if not unk_func or not unk_func(e):
        self.entry_to_index[e] = cur_ind
        cur_ind += 1
      else:
        self.entry_to_index[e] = 0  # 0 represents UNK.

    self.num_embeddings = cur_ind
    # Embedding matrix
    self.emb_mat = theano.shared(
        name='lexicon_emb_mat',
        value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.num_embeddings, emb_size)).astype(theano.config.floatX))

  def get_entries(self, in_str):
    if in_str not in self.entry_map: return []
    return self.entry_map[in_str]

  def size(self):
    return len(self.entries)

  def get_num_embeddings(self):
    return self.num_embeddings

  def get_theano_embedding(self, index):
    return self.emb_mat[index]

  def get_theano_params(self):
    return [self.emb_mat]

  def get_theano_all(self):
    return [self.emb_mat]

  def get_index(self, entry):
    # Note: indices are NOT unique!  Some indices may map to 0 for UNK
    return self.entry_to_index[entry]

  def add_entry(self, entry):
    """Add an entry to the lexicon.  
    
    Always maps to UNK, since embedding matrix is fixed after lexicon creation.
    """
    if entry in self.entry_to_index: return
    self.entries.append(entry)
    self.entry_to_index[entry] = 0  # 0 is for UNK
    self.entry_map[entry[0]].append(entry)

  @classmethod
  def from_sentences(cls, sentences, emb_size, unk_cutoff):
    """Create a lexicon that just maps every word in a vocabulary to itself."""
    counts = collections.Counter()
    words = set()
    for s in sentences:
      counts.update(s.split(' '))
      words.update(s.split(' '))
    entries = [(w, w) for w in words]
    lex = cls(entries, emb_size, unk_func=lambda e: counts[e[0]] <= unk_cutoff)
    print 'Extract lexicon of size %d, %d embeddings' % (
        lex.size(), lex.get_num_embeddings())
    return lex
