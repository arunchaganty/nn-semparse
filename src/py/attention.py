"""A soft attention model

We use the global attention model with input feeding
used by Luong et al. (2015).
See http://stanford.edu/~lmthang/data/papers/emnlp15_attn.pdf
"""
import itertools
import numpy
import theano
from theano import tensor as T
import sys

from attnspec import AttentionSpec
from derivation import Derivation
from neural import NeuralModel
from vocabulary import Vocabulary

class AttentionModel(NeuralModel):
  """An encoder-decoder RNN model."""
  def setup(self):
    self.setup_encoder()
    self.setup_decoder_step()
    self.setup_decoder_write()
    self.setup_backprop()

  @classmethod
  def get_spec_class(cls):
    return AttentionSpec

  def _symb_encoder(self, x):
    """The encoder (symbolically), for decomposition."""
    def fwd_rec(x_t, h_prev, *params):
      return self.spec.f_enc_fwd(x_t, h_prev)
    def bwd_rec(x_t, h_prev, *params):
      return self.spec.f_enc_bwd(x_t, h_prev)

    fwd_states, _ = theano.scan(fwd_rec, sequences=[x],
                                outputs_info=[self.spec.get_init_fwd_state()],
                                non_sequences=self.spec.get_all_shared())
    bwd_states, _ = theano.scan(bwd_rec, sequences=[x],
                                outputs_info=[self.spec.get_init_bwd_state()],
                                non_sequences=self.spec.get_all_shared(),
                                go_backwards=True)
    enc_last_state = T.concatenate([fwd_states[-1], bwd_states[-1]])
    dec_init_state = self.spec.get_dec_init_state(enc_last_state)

    bwd_states = bwd_states[::-1]  # Reverse backward states.
    def concat_rec(h_fwd, h_bwd, *params):
      return T.concatenate([h_fwd, h_bwd])
    annotations, _ = theano.scan(concat_rec, sequences=[fwd_states, bwd_states],
                                 non_sequences=self.spec.get_all_shared())
    return (dec_init_state, annotations)

  def setup_encoder(self):
    """Run the encoder.  Used at test time."""
    x = T.lvector('x_for_enc')
    dec_init_state, annotations = self._symb_encoder(x)
    self._encode = theano.function(
        inputs=[x], outputs=[dec_init_state, annotations])

  def setup_decoder_step(self):
    """Advance the decoder by one step.  Used at test time."""
    y_t = T.lscalar('y_t_for_dec')
    c_prev = T.vector('c_prev_for_dec')
    h_prev = T.vector('h_prev_for_dec')
    h_t = self.spec.f_dec(y_t, c_prev, h_prev)
    self._decoder_step = theano.function(inputs=[y_t, c_prev, h_prev], outputs=h_t)

  def setup_decoder_write(self):
    """Get the write distribution of the decoder.  Used at test time."""
    annotations = T.matrix('annotations_for_write')
    h_prev = T.vector('h_prev_for_write')
    cur_lex_entries = T.lvector('cur_lex_entries_for_write')
    h_for_write = self.spec.decoder.get_h_for_write(h_prev)
    scores = self.spec.get_attention_scores(h_for_write, annotations)
    alpha = self.spec.get_alpha(scores)
    c_t = self.spec.get_context(alpha, annotations)
    write_dist = self.spec.f_write(h_for_write, c_t, cur_lex_entries, scores)
    self._decoder_write = theano.function(inputs=[annotations, h_prev, cur_lex_entries],
                                          outputs=[write_dist, c_t, alpha],
                                          on_unused_input='warn')  # For lexicon

  def setup_backprop(self):
    x = T.lvector('x_for_backprop')
    y = T.lvector('y_for_backprop')
    cur_lex_entries = T.lvector('cur_lex_entries_for_backprop')
    y_lex_inds = T.lmatrix('y_lex_inds_for_backprop')
    y_in_x_inds = T.lmatrix('y_in_x_inds_for_backprop')
    dec_init_state, annotations = self._symb_encoder(x)

    def decoder_recurrence(y_t, cur_y_lex_inds, cur_y_in_x_inds, h_prev,
                           annotations, cur_lex_entries, *params):
      h_for_write = self.spec.decoder.get_h_for_write(h_prev)
      scores = self.spec.get_attention_scores(h_for_write, annotations)
      alpha = self.spec.get_alpha(scores)
      c_t = self.spec.get_context(alpha, annotations)
      write_dist = self.spec.f_write(h_for_write, c_t, cur_lex_entries, scores)
      base_p_y_t = write_dist[y_t] 
      if self.spec.attention_copying:
        copying_p_y_t = T.dot(
            write_dist[self.out_vocabulary.size():],
            cur_y_in_x_inds)
      else:
        copying_p_y_t = T.dot(
            write_dist[self.out_vocabulary.size():],
            cur_y_lex_inds)
      p_y_t = base_p_y_t + copying_p_y_t
      h_t = self.spec.f_dec(y_t, c_t, h_prev)
      return (h_t, p_y_t)

    dec_results, _ = theano.scan(
        fn=decoder_recurrence, sequences=[y, y_lex_inds, y_in_x_inds],
        outputs_info=[dec_init_state, None],
        non_sequences=[annotations, cur_lex_entries] + self.spec.get_all_shared())
    p_y_seq = dec_results[1]
    log_p_y = T.sum(T.log(p_y_seq))
    gradients = T.grad(log_p_y, self.params)
    self._backprop = theano.function(
        inputs=[x, y, cur_lex_entries, y_lex_inds, y_in_x_inds],
        outputs=[p_y_seq, log_p_y] + [-g for g in gradients])

  def decode_greedy(self, ex, max_len=100):
    # TODO: make this have the same interface as decode_beam
    h_t, annotations = self._encode(ex.x_inds)
    y_tok_seq = []
    p_y_seq = []  # Should be handy for error analysis
    p = 1
    for i in range(max_len):
      write_dist, c_t, alpha = self._decoder_write(annotations, h_t, ex.lex_inds)
      y_t = numpy.argmax(write_dist)
      p_y_t = write_dist[y_t]
      p_y_seq.append(p_y_t)
      p *= p_y_t
      if y_t == Vocabulary.END_OF_SENTENCE_INDEX:
        break
      if y_t < self.out_vocabulary.size():
        y_tok = self.out_vocabulary.get_word(y_t)
      else:
        new_ind = y_t - self.out_vocabulary.size()
        if self.spec.attention_copying:
          y_tok = ex.x_toks[new_ind]
        else:
          lex_entry = ex.lex_entries[new_ind]
          y_tok = lex_entry[1]
        y_t = self.out_vocabulary.get_index(y_tok)
      y_tok_seq.append(y_tok)
      h_t = self._decoder_step(y_t, c_t, h_t)
    return [(p, y_tok_seq)]

  def decode_beam(self, ex, beam_size=1, max_len=100):
    h_t, annotations = self._encode(ex.x_inds)
    beam = [[Derivation(ex, 1, [], hidden_state=h_t, 
                        attention_list=[], copy_list=[])]]
    finished = []
    for i in range(1, max_len):
      new_beam = []
      for deriv in beam[i-1]:
        cur_p = deriv.p
        h_t = deriv.hidden_state
        y_tok_seq = deriv.y_toks
        attention_list = deriv.attention_list
        copy_list = deriv.copy_list
        write_dist, c_t, alpha = self._decoder_write(annotations, h_t, ex.lex_inds)
        sorted_dist = sorted([(p_y_t, y_t) for y_t, p_y_t in enumerate(write_dist)],
                             reverse=True)
        for j in range(beam_size):
          p_y_t, y_t = sorted_dist[j]
          new_p = cur_p * p_y_t
          if y_t == Vocabulary.END_OF_SENTENCE_INDEX:
            finished.append(Derivation(ex, new_p, y_tok_seq,
                                       attention_list=attention_list + [alpha],
                                       copy_list=copy_list + [0]))
            continue
          if y_t < self.out_vocabulary.size():
            y_tok = self.out_vocabulary.get_word(y_t)
            do_copy = 0
          else:
            new_ind = y_t - self.out_vocabulary.size()
            augmented_x_toks = ex.x_toks + [Vocabulary.END_OF_SENTENCE]
            if self.spec.attention_copying:
              y_tok = augmented_x_toks[new_ind]
            else:
              lex_entry = ex.lex_entries[new_ind]
              y_tok = lex_entry[1]
            y_t = self.out_vocabulary.get_index(y_tok)
            do_copy = 1
          new_h_t = self._decoder_step(y_t, c_t, h_t)
          new_entry = Derivation(ex, new_p, y_tok_seq + [y_tok],
                                 hidden_state=new_h_t,
                                 attention_list=attention_list + [alpha],
                                 copy_list=copy_list + [do_copy])
          new_beam.append(new_entry)
      new_beam.sort(key=lambda x: x.p, reverse=True)
      beam.append(new_beam[:beam_size])
    return sorted(finished, key=lambda x: x.p, reverse=True)
