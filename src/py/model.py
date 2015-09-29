"""An interspersed read-write neural network model."""
import collections
import numpy
import sys
import theano
from theano.ifelse import ifelse
from theano import tensor as T
import os

from neural import NeuralModel
from vocabulary import Vocabulary

CLIP_THRESH = 3.0  # Clip gradient if norm is larger than this

class IRWModel(NeuralModel):
  """Abstract class for an interspersed read-write neural network.

  This class contains some basic outer functions as well as some
  helper functions that are expected to be of general use.
  
  See writeup in tex/irw-doc.pdf for details.
  We deviate in a couple ways:
  - r_t and w_t here are integers, not vectors.  They represent the index
    in the vocabulary of the word being read or written,
    or -1 to indicate "None".

  Concrete sublcasses must implement the following methods:
  - self.setup(): set up necessary theano functions
  - self.get_objective_and_gradients(x_inds, y_inds):
        get (estimate of) (objective, gradients w.r.t. params)
  """

  def get_score(self, x_inds, y_inds):
    # TODO: this
    return self._get_prob(x_inds, y_inds)

  def setup_map(self):
    # Index (in vocabulary) of input and output words
    x = T.lvector('x_for_map')
    output_len = T.lscalar('output_len_for_map')

    # Compute (greedy, approximate) MAP, for decoding
    def recurrence_map(i, r_t, w_t, h_t, next_read):
      # Force a read at the first step, don't smooth
      p_r = ifelse(T.eq(i, 0), self.float_type(1.0), self.spec.f_p_read(h_t))
      p_dist_w = self.spec.f_dist_write(h_t)
      write_candidate = T.argmax(p_dist_w)
      p_w = p_dist_w[write_candidate]

      # Read iff p_r > .5 and there are more words to read
      do_read = T.gt(p_r, .5) & T.lt(next_read, x.shape[0])

      r_next = ifelse(do_read, x[next_read], numpy.int64(-1))
      w_next = ifelse(do_read, numpy.int64(-1), T.argmax(p_dist_w))
      h_next = self.spec.f_rnn(r_next, w_next, h_t)
      p = ifelse(do_read, p_r, (1-p_r) * p_w)
      read_index = ifelse(do_read, next_read + 1, next_read)

      return (r_next, w_next, h_next, p, read_index)

    results, _ = theano.scan(
        fn=recurrence_map,
        sequences=T.arange(x.shape[0] + output_len),
        outputs_info=[numpy.int64(-1), numpy.int64(-1), self.spec.h0, None, numpy.int64(0)])
    r = results[0]
    w = results[1]
    self._get_map = theano.function(inputs=[x, output_len], outputs=[r, w])

  def setup_step(self):
    r_t = T.lscalar('r_t_for_step')
    w_t = T.lscalar('w_t_for_step')
    h_prev = T.vector('h_prev_for_step')

    h_t = self.spec.f_rnn(r_t, w_t, h_prev)
    p_r = self.spec.f_p_read(h_t)
    p_dist_w = self.spec.f_dist_write(h_t)

    self._step_forward = theano.function(
        inputs=[r_t, w_t, h_prev],
        outputs=[h_t, p_r, p_dist_w])

  def setup_regularization(self):
    lambda_reg = T.scalar('lambda_for_regularization')
    reg_val = self.spec.get_regularization(lambda_reg)
    if reg_val:
      reg_gradients = T.grad(reg_val, self.params, disconnected_inputs='ignore')
      self._get_regularization_info = theano.function(
          inputs=[lambda_reg], outputs=[reg_val] + reg_gradients)
    else:
      self.get_regularization_info = (
          lambda lambda_reg: numpy.zeros(len(self.params) + 1))

  def decode_greedy(self, x, max_len=100):
    r, w = self._get_map(x, max_len)
    r_list = list(r)
    w_list = list(w)
    try:
      eos_ind = w_list.index(Vocabulary.END_OF_SENTENCE_INDEX)
    except ValueError:
      eos_ind = len(w_list) - 1
    r_out = r_list[:(eos_ind+1)]
    w_out = w_list[:(eos_ind+1)]
    return r_out, w_out

  def decode_beam(self, x, max_len=100, beam_size=5):
    print 'decode_beam'
    BeamState = collections.namedtuple(
        'BeamState', ['r_seq', 'w_seq', 'h_prev', 'next_read', 'log_p'])
    best_finished_state = None
    max_log_p = float('-Inf')
    beam = []
    # Start with a read
    beam.append([BeamState([x[0]], [-1], self.spec.h0.get_value(), 1, 0)])
    for i in range(1, max_len):
      candidates = []
      for state in beam[i-1]:
        if state.w_seq[-1] == Vocabulary.END_OF_SENTENCE_INDEX:
          if state.log_p > max_log_p:
            max_log_p = state.log_p
            best_finished_state = state
          continue
        if state.log_p < max_log_p: continue  # Prune here
        h_t, p_r, p_dist_w = self._step_forward(
            state.r_seq[-1], state.w_seq[-1], state.h_prev)
        if state.next_read < len(x):
          read_state = BeamState(
              state.r_seq + [x[state.next_read]], state.w_seq + [-1], h_t,
              state.next_read + 1, state.log_p + numpy.log(p_r))
          candidates.append(read_state)
        else:
          p_r = 0  # Force write
        if p_r < 1:
          write_candidates = sorted(enumerate(p_dist_w), key=lambda x: x[1],
                                    reverse=True)[:beam_size]
          for index, prob in write_candidates:
            new_state = BeamState(
                state.r_seq + [-1], state.w_seq + [index], h_t, state.next_read, 
                state.log_p + numpy.log(1 - p_r) + numpy.log(prob))
            candidates.append(new_state)
      beam.append(sorted(
          candidates, key=lambda x: x.log_p, reverse=True)[:beam_size])

    return (best_finished_state.r_seq, best_finished_state.w_seq)

  def get_gradient_seq(self, y_seq):
    """Compute gradient with respect to a sequence."""
    def grad_fn(j, y, *params):
      return T.grad(y[j], self.params, disconnected_inputs='warn')
    results, _ = theano.scan(fn=grad_fn,
                             sequences=T.arange(y_seq.shape[0]),
                             non_sequences=[y_seq] + self.params,
                             strict=True)
    # results[i][j] is gradient of y[j] w.r.t. self.params[i]
    return results

  def perform_gd_step(self, param, gradient, eta):
    """Do a gradient descent step."""
    #print param.name
    #print param.get_value()
    #print gradient
    old_value = param.get_value()
    grad_norm = numpy.sqrt(numpy.sum(gradient**2))
    if grad_norm >= CLIP_THRESH:
      gradient = gradient * CLIP_THRESH / grad_norm
      new_norm = numpy.sqrt(numpy.sum(gradient**2))
      print 'Clipped norm of %s from %g to %g' % (param, grad_norm, new_norm)
    new_value = old_value - eta * gradient
    param.set_value(new_value)

  def on_train_epoch(self, t):
    for p in self.params:
      print '%s: %s' % (p.name, p.get_value())
