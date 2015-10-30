"""A single example in a dataset."""
class Example(object):
  """A single example in a dataset.

  Basically a struct after it's initialized, with the following fields:
    - self.x_str, self.y_str: input/output as single space-separated strings
    - self.x_toks, self.y_toks: input/output as list of strings
    - self.input_vocab, self.output_vocab: Vocabulary objects
    - self.x_inds, self.y_inds: input/output as indices in corresponding vocab
    - self.y_input_inds: for each i, a bit vector of length len(self.x)
          where j-th bit is true iff x[j] == y[i].

  Treat these objects as read-only.
  """
  def __init__(self, x_str, y_str, input_vocab, output_vocab, reverse_input=False):
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
    self.y_input_inds = [[int(x_j == y_i) for x_j in self.x_toks] + [0]
                         for y_i in self.y_toks] + [[0] * len(self.x_inds)]
        # Add 0's for the EOS tag in both x and y
