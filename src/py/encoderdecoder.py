"""A basic encoder-decoder model."""
import itertools
import numpy
import theano
from theano import tensor as T
import sys

from neural import NeuralModel
from vocabulary import Vocabulary

class EncoderDecoderModel(NeuralModel):
  """An encoder-decoder RNN model."""
  def setup(self):
    self.setup_encoder()
    self.setup_decoder_step()
    self.setup_decoder_write()
    self.setup_backprop()

  def setup_encoder(self):
    """Run the encoder.  Used at test time."""
    x = T.lvector('x_for_enc')
    def recurrence(x_t, h_prev, *params):
      return self.spec.f_enc(x_t, h_prev)
    results, _ = theano.scan(recurrence,
                             sequences=[x],
                             outputs_info=[self.spec.get_init_state()],
                             non_sequences=self.spec.get_all_shared())
    h_last = results[-1]
    self._encode = theano.function(inputs=[x], outputs=h_last)

  def setup_decoder_step(self):
    """Advance the decoder by one step.  Used at test time."""
    y_t = T.lscalar('y_t_for_dec')
    h_prev = T.vector('h_prev_for_dec')
    h_t = self.spec.f_dec(y_t, h_prev)
    self._decoder_step = theano.function(inputs=[y_t, h_prev], outputs=h_t)

  def setup_decoder_write(self):
    """Get the write distribution of the decoder.  Used at test time."""
    h_prev = T.vector('h_prev_for_write')
    cur_lex_entries = T.lvector('cur_lex_entries_for_write')
    h_for_write = self.spec.decoder.get_h_for_write(h_prev)
    write_dist = self.spec.f_write(h_for_write, cur_lex_entries)
    self._decoder_write = theano.function(inputs=[h_prev, cur_lex_entries], outputs=write_dist)

  def setup_backprop(self):
    x = T.lvector('x_for_backprop')
    y = T.lvector('y_for_backprop')
    cur_lex_entries = T.lvector('cur_lex_entries_for_backprop')
    y_input_inds = T.lmatrix('y_input_inds_for_backprop')
    def enc_recurrence(x_t, h_prev, *params):
      return self.spec.f_enc(x_t, h_prev)
    enc_results, _ = theano.scan(fn=enc_recurrence,
                                 sequences=[x],
                                 outputs_info=[self.spec.get_init_state()],
                                 non_sequences=self.spec.get_all_shared())
    h_last = enc_results[-1]
    
    def decoder_recurrence(y_t, cur_y_input_inds, h_prev, cur_lex_entries, *params):
      h_for_write = self.spec.decoder.get_h_for_write(h_prev)
      write_dist = self.spec.f_write(h_for_write, cur_lex_entries)
      p_y_t = write_dist[y_t] + T.dot(
          write_dist[self.out_vocabulary.size():],
          cur_y_input_inds)

      h_t = self.spec.f_dec(y_t, h_prev)
      return (h_t, p_y_t)
    dec_results, _ = theano.scan(
        fn=decoder_recurrence, sequences=[y, y_input_inds],
        outputs_info=[h_last, None],
        non_sequences=[cur_lex_entries] + self.spec.get_all_shared())
    p_y_seq = dec_results[1]
    log_p_y = T.sum(T.log(p_y_seq))
    gradients = T.grad(log_p_y, self.params)
    self._backprop = theano.function(
        inputs=[x, y, cur_lex_entries, y_input_inds],
        outputs=[p_y_seq, log_p_y] + [-g for g in gradients])

  def get_objective_and_gradients(self, ex):
    # TODO(robinjia): Only pass x for cur_lex_entries if lexicon is simple.
    info = self._backprop(ex.x_inds, ex.y_inds, ex.x_inds, ex.y_input_inds)
    p_y_seq = info[0]
    log_p_y = info[1]
    gradients_list = info[2:]
    objective = -log_p_y
    gradients = dict(itertools.izip(self.params, gradients_list))
    print 'P(y_i): %s' % p_y_seq
    return (objective, gradients)

  def decode_greedy(self, x, max_len=100):
    h_t = self._encode(x)
    y_seq = []
    p_y_seq = []  # Should be handy for error analysis
    for i in range(max_len):
      # TODO(robinjia): only pass in x if lexicon is simple
      write_dist = self._decoder_write(h_t, x)
      y_t = numpy.argmax(write_dist)
      p_y_t = write_dist[y_t]
      y_seq.append(y_t)
      p_y_seq.append(p_y_t)
      if y_t == Vocabulary.END_OF_SENTENCE_INDEX:
        break
      if y_t >= self.out_vocabulary.size():
        # TODO(robinjia): Check if this word is in output dictionary
        y_t = Vocabulary.UNKNOWN_INDEX
      h_t = self._decoder_step(y_t, h_t)
    return y_seq
