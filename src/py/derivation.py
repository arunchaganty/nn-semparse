"""A full or partial derivation."""
class Derivation(object):
  def __init__(self, example, p, y_toks, hidden_state=None,
               attention_list=None):
    self.example = example
    self.p = p
    self.y_toks = y_toks
    self.hidden_state = hidden_state
    self.attention_list = attention_list
