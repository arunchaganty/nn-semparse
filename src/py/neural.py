"""Generic class for neural net model."""
import random
import sys
import time

class NeuralModel:
  """A generic continuous neural model for paraphrase.

  Implementing classes must implement the following functions:
  - self.train_batch(examples, eta=0.1, T=10, verbose=False): mini-batch SGD update.
  - self.get_score(x_inds, y_inds): get score for candidate phrase pair.
  """
  def __init__(self, spec, float_type=numpy.float64):
    """Initialize.

    Args:
      spec: IRWSpec object.
      float_type: Floating point type (default 64-bit/double precision)

    Convention used by this class:
      nh: dimension of hidden layer
      nw: number of words in the vocabulary
      de: dimension of word embeddings
    """
    self.spec = spec
    self.float_type = float_type

    self.in_vocabulary = spec.in_vocabulary
    self.de_in = spec.de_in
    self.nw_in = spec.nw_in

    self.out_vocabulary = spec.out_vocabulary
    self.de_out = spec.de_out
    self.nw_out = spec.nw_out

    self.de_total = spec.de_total
    self.nh = spec.nh
    self.params = spec.get_params()
    self.all_shared = spec.get_all_shared()

    self.setup_map()
    self.setup_step()
    self.setup_regularization()
    self.setup()
    print >> sys.stderr, 'Setup complete.'

  def setup(self):
    raise NotImplementedError

  def get_objective_and_gradients(self, x, y, **kwargs):
    """Get objective and gradients.

    Returns: tuple (objective, gradients) where
      objective: the current objective value
      gradients: map from parameter to gradient
    """
    raise NotImplementedError

  def train_batch(self, examples, eta, do_update=True):
    """Run training given a batch of training examples.
    
    Returns negative log-likelihood.
    If do_update is False, compute nll but don't do the gradient step.
    """
    objective = 0
    gradients = {}
    for ex in examples:
      x_inds, y_inds, kwargs = ex
      print 'x: %s' % self.in_vocabulary.indices_to_sentence(x_inds)
      print 'y: %s' % self.out_vocabulary.indices_to_sentence(y_inds)
      cur_objective, cur_gradients = self.get_objective_and_gradients(
          x_inds, y_inds, total_examples=len(examples), **kwargs)
      objective += cur_objective
      for p in self.params:
        if p in gradients:
          gradients[p] += cur_gradients[p] / len(examples)
        else:
          gradients[p] = cur_gradients[p] / len(examples)
    if do_update:
      for p in self.params:
        self.perform_gd_step(p, gradients[p], eta)
    return objective

  def get_score(self, x_inds, y_inds):
    """Should return a score for a candidate paraphrase pair.
    
    Usually good idea to return a probability.
    """
    raise NotImplementedError

  def on_train_epoch(self, t):
    """Optional method to do things every epoch."""
    pass

  def train(self, dataset, eta=0.1, T=10, verbose=False, batch_size=1):
    # batch_size = size for mini batch.  Defaults to SGD.
    for it in range(T):
      t0 = time.time()
      total_nll = 0
      random.shuffle(dataset)
      for i in range(0, len(dataset), batch_size):
        do_update = i + batch_size <= len(dataset)
        cur_examples = dataset[i:(i+batch_size)]
        nll = self.train_batch(cur_examples, eta, do_update=do_update)
        total_nll += nll
        if verbose:
          print 'NeuralModel.train(): iter %d, example = %s: nll = %g' % (
              it, str(ex), nll)
      t1 = time.time()
      print 'NeuralModel.train(): iter %d: total nll = %g (%g seconds)' % (
          it, total_nll, t1 - t0)
      self.on_train_epoch(it)

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
  def test(self, dataset, sdf_dataset=None):
    """Evaluate on an SDF-structured dataset, doing re-ranking.
    
    Expects dataset to be formatted like output of sdf_to_test_dataset()."""
    num_true = 0
    for i, ex in enumerate(dataset):
      x_inds = ex.utterance
      scores = [self.get_score(x_inds, c[0]) for c in ex.candidates]
      argmax = max(enumerate(scores), key=lambda x: x[1])[0]
      #print sorted(zip(scores, list(c[1] for c in ex.candidates)),
      #             key=lambda x: x[0], reverse=True)[:5]
      if ex.candidates[argmax][1]:
        num_true += 1
      if sdf_dataset:
        records = sdf_dataset[i]
        candidates = sorted(zip(scores, records), key=lambda x: x[0], reverse=True)
        score_predicted = candidates[0][0]
        pred_record = candidates[0][1]
        utterance = pred_record.utterance
        predicted = pred_record.canonical_utterance
        all_correct = [c for c in candidates if c[1].compatibility]
        if all_correct:
          score_best = all_correct[0][0]
          best = all_correct[0][1].canonical_utterance
          best_rank = [c[0] for c in candidates].index(score_best)
        else:
          best = 'NONE'
          score_best = float('NaN')
          best_rank = -1
        print 'Example %d:' % i
        print '  Utterance: "%s"' % utterance
        print '  Predicted: "%s" (score = %g)' % (predicted, score_predicted)
        print '  Best Correct: "%s" (score = %g, rank = %d / %d)' % (
            best, score_best, best_rank, len(candidates))
        print '  Correct: %s' % bool(pred_record.compatibility)

    print 'Accuracy: %d/%d = %g' % (
        num_true, len(dataset), float(num_true) / len(dataset))

  def sdf_to_test_data(self, sdf_data):
    """Convert SDF dataset into one fit for testing the model.

    Test dataset format is list of examples, where each example
    is has the following structure:
    example.utterance: list of indices in the vocabulary
    example.candidates: list of pairs (x, y) where
      - x is a list of indices in the vocabulary
      - y is 1 or 0 indicating correct/incorrect
    """
    test_data = []
    for records in sdf_data:
      utterance = records[0].utterance
      candidates = []
      for record in records:
        candidates.append(
            (self.vocabulary.sentence_to_indices(record.canonical_utterance),
             bool(record.compatibility)))
      x_inds = self.vocabulary.sentence_to_indices(utterance)
      if self.reverse_source:
        x_inds = x_inds[::-1]
      example = TestExample(
          utterance=x_inds,
          candidates=candidates)
      test_data.append(example)
    return test_data
