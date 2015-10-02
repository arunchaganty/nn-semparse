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
  """IRW model with sequence-based objective and variational gradient.

  Objective is the standard \log P(y).

  We can approximate this by sampling from a proposal distribution q(z | x, y),
  where z denotes the order of reads and writes 
  (i.e. the "alignment" between x  and y).
  This proposal distribution is sampled with sequential monte carlo,
  where we force the write of y_i at the i-th write action,
  but use the model's own probabilities of reading vs. writing.
  """
  def setup(self, float_type=numpy.float64):
    self.setup_sample()
    self.setup_map_alignment()
    self.setup_infer()

    # For epsilon-greedy exploration
    self.epsilon = 0.0  
    self.epsilon_max = 0.0
    self.num_exploration_iters = 0

  def set_epsilon_max(self, eps_max):
    """Set epsilon for epsilon-greedy exploration."""
    self.epsilon_max = eps_max

  def set_epsilon_fixed(self, eps, num_iters):
    """Set a fixed epsilon for epsilon-greedy.
    
    Args:
      eps: The probability of taking a random action (read/write) 
      num_iters: Number of epochs to use this for (afterwards, 0)
    """
    self.epsilon = eps
    self.num_exploration_iters = num_iters

  def smoothed_f_p_read(self, h_t, epsilon):
    """Smooths probability of read to favor a uniform distribution over alignments."""
    p_raw = self.spec.f_p_read(h_t)

    # Epsilon-greedy strategy.
    p_smoothed = epsilon / 2 + (1 - epsilon) * p_raw 
    return p_smoothed

  def setup_sample(self):
    """Sample the alignment from the proposal distribution."""
    # Index (in vocabulary) of input and output words
    x = T.lvector('x_for_sample')
    y = T.lvector('y_for_sample')

    # Epsilon for epsilon-greedy exploration
    epsilon = T.scalar('eps_for_sample')

    # Random number generator
    self.rng = RandomStreams(1234)  # Deterministic seed

    # Sample via sequential monte carlo
    def recurrence_sample(i, r_t, w_t, h_t, next_read, next_write, my_x, my_y,
                          my_eps, *all_shared):
      # Force a read at the first step
      p_r = ifelse(T.eq(i, 0), self.float_type(1.0),
                   self.smoothed_f_p_read(h_t, my_eps))
      choose_read = self.rng.binomial(p=p_r)
      has_next_read = T.lt(next_read, my_x.shape[0])
      has_next_write = T.lt(next_write, my_y.shape[0])

      # We use bitwise ops to do logic
      a_t = ((1 - has_next_write) |
             (T.eq(choose_read, 1) & has_next_read))

      r_next = ifelse(a_t, my_x[next_read], numpy.int64(-1))
      w_next = ifelse(a_t, numpy.int64(-1), my_y[next_write])
      h_next = self.spec.f_rnn(r_next, w_next, h_t)

      read_index = ifelse(a_t, next_read + 1, next_read)
      write_index = ifelse(a_t, next_write, next_write + 1)

      return (r_next, w_next, h_next, read_index, write_index)

    results, updates = theano.scan(
        fn=recurrence_sample,
        sequences=T.arange(x.shape[0] + y.shape[0]),
        outputs_info=[numpy.int64(-1), numpy.int64(-1), self.spec.h0, 
                      numpy.int64(0), numpy.int64(0)],
        non_sequences=[x, y, epsilon] + self.all_shared,
        strict=True)
    r = results[0]
    w = results[1]
    self._get_sample = theano.function(inputs=[x, y, epsilon], outputs=[r, w],
                                       updates=updates,
                                       allow_input_downcast=True)

  def setup_map_alignment(self):
    """Similar to above, just get the MAP alignment."""
    # Index (in vocabulary) of input and output words
    x = T.lvector('x_for_sample')
    y = T.lvector('y_for_sample')

    def recurrence_map(i, r_t, w_t, h_t, next_read, next_write, my_x, my_y,
                       *all_shared):
      # Force a read at the first step
      p_r = ifelse(T.eq(i, 0), self.float_type(1.0),
                   self.smoothed_f_p_read(h_t, 0.0))
      choose_read = ifelse(T.gt(p_r, 0.5), 1, 0)
      has_next_read = T.lt(next_read, my_x.shape[0])
      has_next_write = T.lt(next_write, my_y.shape[0])

      # We use bitwise ops to do logic
      a_t = ((1 - has_next_write) |
             (T.eq(choose_read, 1) & has_next_read))

      r_next = ifelse(a_t, my_x[next_read], numpy.int64(-1))
      w_next = ifelse(a_t, numpy.int64(-1), my_y[next_write])
      h_next = self.spec.f_rnn(r_next, w_next, h_t)

      read_index = ifelse(a_t, next_read + 1, next_read)
      write_index = ifelse(a_t, next_write, next_write + 1)

      return (r_next, w_next, h_next, read_index, write_index)

    results, _ = theano.scan(
        fn=recurrence_map,
        sequences=T.arange(x.shape[0] + y.shape[0]),
        outputs_info=[numpy.int64(-1), numpy.int64(-1), self.spec.h0, 
                      numpy.int64(0), numpy.int64(0)],
        non_sequences=[x, y] + self.all_shared,
        strict=True)
    r = results[0]
    w = results[1]
    self._get_map_alignment = theano.function(inputs=[x, y], outputs=[r, w])

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

  def get_samples(self, x, y, epsilon, num_samples):
    # Make them hashable to be nice
    def make_hashable(sample):
      r, w = sample
      return (tuple(r), tuple(w))
    return [make_hashable(self._get_sample(x, y, epsilon)) for i in range(num_samples)]

  def compute_epsilon(self, x, y):
    if self.num_exploration_iters > 0 and self.epsilon > 0:
      # Fixed epsilon
      epsilon = self.epsilon
    elif self.epsilon_max > 0:
      # Epsilon depends on current MAP alignemnt.
      r_map, w_map = self._get_map_alignment(x, y)
      log_perplexity = self._get_log_perplexity(r_map, w_map, x, y, 0.0)
      perplexity = numpy.exp(log_perplexity)
      epsilon = self.epsilon_max * (1 - perplexity)
    else:
      epsilon = 0.0

    # Some helpful debugging output
    print 'epsilon: %g' % epsilon
    if numpy.isnan(epsilon):
      print 'r_map: %s' % r_map
      print 'w_map: %s' % w_map
      print 'log_perplexity: %g' % log_perplexity
      print 'perplexity: %g' % perplexity
      for p in self.params:
        print p
        print p.get_value()

    return epsilon

  def on_train_epoch(self, t):
    IRWModel.on_train_epoch(self, t)
    self.num_exploration_iters -= 1

  def get_objective_and_gradients(self, x, y, lambda_reg=0.01, num_samples=50,
                                  total_examples=1, actions=None):
    epsilon = self.compute_epsilon(x, y)
    samples = self.get_samples(x, y, epsilon, num_samples)
    sample_counter = collections.Counter(samples)

    map_sample, count = sample_counter.most_common(1)[0]
    print 'map_sample: r = %s, w = %s, count = %d' % (map_sample[0], map_sample[1], count)

    objective = self.float_type(0.0)
    gradients = {}
    for p in self.params:
      gradients[p] = numpy.zeros_like(p.get_value())
    # Probability of writing y_i, on average
    write_probs = numpy.zeros(len(y))

    for s in sample_counter:
      r, w = s
      count = sample_counter[s]
      weight = float(count) / len(samples)

      info = self._get_info(r, w, x, y, epsilon)
      cur_log_q = info[0]
      cur_log_p_y = info[1]
      cur_p_y_seq = info[2]
      cur_gradients = info[3:]

      # Compute the objective, approximately
      # It's easier just to compute the variational approximation,
      # which is just E[\log(p(y)) - \log(q(z))]
      # Recall that the objective is negative log likelihood,
      # hence a positive number
      objective -= weight * (cur_log_p_y - cur_log_q)

      # Get gradient by averaging over sample
      saw_non_finite = False
      for p, g in itertools.izip(self.params, cur_gradients):
        if numpy.isfinite(g).all():
          gradients[p] += weight * g
        else:
          # Gradient update had NaN
          has_nan = numpy.isnan(g).any()
          has_inf = numpy.isposinf(g).any()
          has_neg_inf = numpy.isneginf(g).any()
          print 'Gradient of %s contains: nan=%s, inf=%s, neginf=%s' % (
              p, has_nan, has_inf, has_neg_inf)
          saw_non_finite = True
      if saw_non_finite:
        print 'WARNING: non-finite gradients in sample with count %d' % count

      # For debugging, use p_y_seq
      cur_write_inds = [i for i in range(len(w)) if w[i] >= 0]
      cur_write_probs = numpy.array([cur_p_y_seq[i] for i in cur_write_inds])
      write_probs += weight * cur_write_probs

    print write_probs

    # Add regularization
    #reg_info = self._get_regularization_info(lambda_reg / total_examples)
    #reg_value = reg_info[0]
    #reg_gradients = reg_info[1:]
    #for p, g in itertools.izip(self.params, reg_gradients):
    #  gradients[p] += g

    for p in self.params:
      print '%s: min = %g, max = %g, grad norm = %g' % (
          p, numpy.min(p.get_value()), numpy.max(p.get_value()),
          numpy.sqrt(numpy.sum(gradients[p] ** 2)))

    return (objective, gradients)
