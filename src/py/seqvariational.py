"""An IRW model with sequence-based objective and variational gradient."""
import collections
import itertools
import numpy
import sys
import theano
from theano.ifelse import ifelse
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from model import IRWModel

class SeqVariationalIRWModel(IRWModel):
  def setup(self, float_type=numpy.float64):
    self.setup_map_alignment()
    self.setup_infer()

  def setup_map_alignment(self):
    """Similar to above, just get the MAP alignment."""
    # Index (in vocabulary) of input and output words
    x = T.lvector('x_for_sample')
    y = T.lvector('y_for_sample')

  def setup_infer(self):
    # Index (in vocabulary) of reads/writes (or -1 if None)
    r = T.lvector('r_for_infer')
    w = T.lvector('w_for_infer')
    x = T.lvector('x_for_infer')
    y = T.lvector('y_for_infer')
    epsilon = T.scalar('eps_for_infer')

    # Compute the log-probability of the sequence r and w.
    def recurrence_prob(r_t, w_t, h_t, next_read, next_write, my_x, my_y,
                        my_eps, *all_shared):
      # Compute P(a_t), i.e. q_t
      a_t = T.ge(r_t, 0)  # 1 for read, 0 for write

      # Force a read at the first step
      p_r = ifelse(T.eq(next_read, 0), self.float_type(1.0),
                   self.smoothed_f_p_read(h_t, my_eps))
      has_next_read = T.lt(next_read, my_x.shape[0])
      has_next_write = T.lt(next_write, my_y.shape[0])
      q_t = ifelse(a_t,
                   # P(read) == 1 if couldn't do more writes.
                   ifelse(has_next_write, p_r, self.float_type(1.0)),  
                   # P(write) == 1 if couldn't do more reads.
                   ifelse(has_next_read, 1 - p_r, self.float_type(1.0)))

      # Compute P(y_i) if we just outputted i-th character
      p_dist_w = self.spec.f_dist_write(h_t)
      p_y_t = T.switch(a_t, self.float_type(1.0), p_dist_w[w_t])

      h_next = self.spec.f_rnn(r_t, w_t, h_t)
      read_index = T.switch(a_t, next_read + 1, next_read)
      write_index = T.switch(a_t, next_write, next_write + 1)

      return (h_next, read_index, write_index, q_t, p_y_t)

    results, _ = theano.scan(fn=recurrence_prob,
                             sequences=[r,w],
                             outputs_info=[self.spec.h0, numpy.int64(0),
                                           numpy.int64(0), None, None],
                             non_sequences=[x, y, epsilon] + self.all_shared,
                             strict=True)

    q_seq = results[3]  # P(a_t)
    p_y_seq = results[4]  # P(y_i)
    log_q = T.sum(T.log(q_seq))
    log_p_y = log_q + T.sum(T.log(p_y_seq))

    # Perplexity of outputting y conditioned on alignment 
    log_perplexity = T.sum(T.log(p_y_seq)) / y.shape[0]
    self._get_log_perplexity = theano.function(inputs=[r, w, x, y, epsilon],
                                               outputs=log_perplexity,
                                               allow_input_downcast=True)

    # To compute gradients, we're using a variational approximation.
    # The likelihood is
    #   \log \sum_z p(y,z) = \log \sum_z q(z)p(y,z)/q(z)
    #                     >= \sum_z q(z) \log(p(y,z)/q(z))
    # Gradient is therefore
    #   \sum_z (\log(p(y,z)/q(z)) q(z) \grad \log q(z) +
    #           q(z) (\grad \log p(y,z) - \grad \log q(z)))
    #  = \E_{z \sim q} (\log(p(y,z)/q(z)) \grad \log q(z) +
    #                   \grad \log p(y,z).
    # Recall that E_{z \sim q}[\grad \log q(z)] = 0.
    #
    # Finally, we negate the gradient since the objective is to
    # minimize negative loglikelihood.
    grad_log_q_all = T.grad(log_q, self.params, disconnected_inputs='warn')
    grad_log_p_y_all = T.grad(log_p_y, self.params, disconnected_inputs='warn')
    gradients = [-(log_p_y - log_q) * grad_log_q - grad_log_p_y
                 for grad_log_q, grad_log_p_y 
                 in itertools.izip(grad_log_q_all, grad_log_p_y_all)]

    self._get_info = theano.function(inputs=[r, w, x, y, epsilon],
                                     outputs=[log_q, log_p_y, p_y_seq] + gradients,
                                     allow_input_downcast=True)
    # For encoder-decoder, only need (negative) gradient w.r.t. p_y
    self._get_enc_dec_info = theano.function(
        inputs=[r, w, x, y, epsilon],
        outputs=[log_q, log_p_y, p_y_seq] + [-g for g in grad_log_p_y_all],
        allow_input_downcast=True)

  def on_train_epoch(self, t):
    IRWModel.on_train_epoch(self, t)
    self.num_exploration_iters -= 1
