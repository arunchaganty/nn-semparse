"""Generic class for neural net model."""
import random
import sys
import time

class NeuralModel:
  """A generic neural model for paraphrase.

  Implementing classes must implement the following functions:
  - self.train_batch(examples, eta=0.1, T=10, verbose=False): mini-batch SGD update.
  - self.get_score(x_inds, y_inds): get score for candidate phrase pair.
  """
  def train_batch(self, examples, eta, do_update=True, **kwargs):
    """How to run training given a batch of training examples.
    
    Should return negative log-likelihood (the objective we're minimizing).
    
    If do_update is False, compute nll but don't do the gradient step.
    """
    raise NotImplementedError

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
