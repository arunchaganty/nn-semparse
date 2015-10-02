"""An IRW model with a fixed alignment."""
import itertools
import numpy
import theano
from theano.ifelse import ifelse
from theano import tensor as T

from seqvariational import SeqVariationalIRWModel

class EncoderDecoderModel(SeqVariationalIRWModel):
  """Encoder-decoder based on the same code as IRW.

  We re-use code extensively from SeqVariationalIRWModel,
  but we don't need to sample because we know the order of
  read and write operations ahead of time.
  """
  def smoothed_f_p_read(self, h_t, epsilon):
    """Don't do any smoothing."""
    return self.spec.f_p_read(h_t)

  def get_enc_dec_alignment(self, x, y):
    r = x + ([-1] * len(y))
    w = ([-1] * len(x)) + y
    return r, w

  def get_objective_and_gradients(self, x, y, **kwargs):
    # Note: we ignore all kwargs such as num_samples, actions, etc.
    r, w = self.get_enc_dec_alignment(x, y)
    print 'r = %s, w = %s' % (r, w)
    info = self._get_enc_dec_info(r, w, x, y, 0.0)
    log_q = info[0]
    log_p_y = info[1]
    p_y_seq = info[2]
    gradients_list = info[3:]
    objective = -log_p_y
    gradients = dict(itertools.izip(self.params, gradients_list))
    print 'P(y_i): %s' % p_y_seq[len(x):]
    return (objective, gradients)

  def setup_map(self):
    """Override setup_map to enforce encoder-decoder ordering."""
    # Index (in vocabulary) of input and output words
    x = T.lvector('x_for_map')
    output_len = T.lscalar('output_len_for_map')

    # Compute (greedy, approximate) MAP, for decoding
    def recurrence_map(i, r_t, w_t, h_t, next_read):
      # Force reads as long as there are more words to read
      do_read = T.lt(next_read, x.shape[0])

      # Choose a word to write
      p_dist_w = self.spec.f_dist_write(h_t)
      write_candidate = T.argmax(p_dist_w)
      p_w = p_dist_w[write_candidate]

      r_next = ifelse(do_read, x[next_read], numpy.int64(-1))
      w_next = ifelse(do_read, numpy.int64(-1), T.argmax(p_dist_w))
      h_next = self.spec.f_rnn(r_next, w_next, h_t)
      p = ifelse(do_read, self.float_type(1.0), p_w)
      read_index = ifelse(do_read, next_read + 1, next_read)

      return (r_next, w_next, h_next, p, read_index)

    results, _ = theano.scan(
        fn=recurrence_map,
        sequences=T.arange(x.shape[0] + output_len),
        outputs_info=[numpy.int64(-1), numpy.int64(-1), self.spec.h0, None, numpy.int64(0)])
    r = results[0]
    w = results[1]
    self._get_map = theano.function(inputs=[x, output_len], outputs=[r, w])

