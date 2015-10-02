"""Classes for dealing with word embeddings."""
import os
import numpy
import theano
import sys

GLOVE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/glove.6B.50d.txt')

class WordEmbeddings:
  """Defines a word embedding interface."""
  def as_theano(self):
    """Return a shared theano matrix, one word per row."""
    raise NotImplementedError

  def to_inds(self, words):
    raise NotImplementedError

  def dimension(self):
    raise NotImplementedError

  def vocab_size(self):
    raise NotImplementedError


class GloveEmbeddings(WordEmbeddings):
  def __init__(self):
    self.word_to_inds = {}
    self.mat = []
    self.dim = None
    with open(GLOVE_FILE) as f:
      for i, line in enumerate(f):
        toks = line.split(' ')
        word = toks[0]
        vec = [float(x) for x in toks[1:]]
        self.word_to_inds[word] = i
        self.mat.append(vec)
        if i == 0:
          self.dim = len(vec)
    self.theano_mat = theano.shared(name='glove_embeddings',
                                    value=numpy.array(self.mat))

  def as_theano(self):
    return self.theano_mat

  def to_inds(self, words, verbose=False):
    inds = []
    for w in words:
      if w.lower() in self.word_to_inds:
        inds.append(self.word_to_inds[w])
      else:
        if verbose:
          print >> sys.stderr, 'Word "%s" not in GloVe dictionary' % w.lower()
    return numpy.matrix(inds).T

  def dimension(self):
    return self.dim

  def vocab_size(self):
    return len(self.word_to_inds)
