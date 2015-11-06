"""A single example in a dataset."""
class Example(object):
  """A single example in a dataset.

  Basically a struct after it's initialized, with the following fields:
    - self.x_str, self.y_str: input/output as single space-separated strings
    - self.x_toks, self.y_toks: input/output as list of strings
    - self.input_vocab, self.output_vocab: Vocabulary objects
    - self.x_inds, self.y_inds: input/output as indices in corresponding vocab
    - self.lex_entries: list of all lexicon entries relevant to this example
    - self.lex_inds: indices in lexicon corresponding to items
    - self.y_lex_inds: for each i, a bit vector of length len(self.lex_entries)
          where j-th bit is true iff lex_entries[j] == y[i].

  Treat these objects as read-only.
  """
  def __init__(self, x_str, y_str, input_vocab, output_vocab, lexicon,
               reverse_input=False):
    """Create an Example object.
    
    Args:
      x_str: Input sequence as a space-separated string
      y_str: Output sequence as a space-separated string
      input_vocab: Vocabulary object for input
      input_vocab: Vocabulary object for output
      reverse_input: If True, reverse the input.
    """
    self.x_str = x_str  # Don't reverse this, used just for printing out
    self.y_str = y_str
    self.x_toks = x_str.split(' ')
    if reverse_input:
      self.x_toks = self.x_toks[::-1]
    self.y_toks = y_str.split(' ')
    self.input_vocab = input_vocab
    self.output_vocab = output_vocab
    self.x_inds = input_vocab.sentence_to_indices(x_str)
    if reverse_input:
      self.x_inds = self.x_inds[::-1]
    self.y_inds = output_vocab.sentence_to_indices(y_str)

    self.lex_entries = [e for t in self.x_toks for e in lexicon.get_entries(t)]
    # TODO(robinjia): handle multi-word input lexicon entries
    self.lex_inds = [lexicon.get_index(e) for e in self.lex_entries]

    self.y_lex_inds = (
        [[int(e[1] == y_i) for e in self.lex_entries] for y_i in self.y_toks] +
        [[0] * len(self.lex_entries)])
        # Add 0's for the EOS tag in y
