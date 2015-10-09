"""Run tests on toy data for IRW models."""
import argparse
import collections
import itertools
import math
import numpy
import random
import sys
import theano
import os

# IRW imports
from encoderdecoder import EncoderDecoderModel
from encdecspec import VanillaEncDecSpec, GRUEncDecSpec, LSTMEncDecSpec
import spec as specutil

# Imports from parent directory
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from vocabulary import GloveVocabulary, RawVocabulary, Vocabulary

CONTINUOUS_MODELS = collections.OrderedDict([
    ('rnn', VanillaEncDecSpec),
    ('gru', GRUEncDecSpec),
    ('lstm', LSTMEncDecSpec),
])

MODELS = collections.OrderedDict([
    ('encoderdecoder', EncoderDecoderModel),
])

VOCAB_TYPES = collections.OrderedDict([
    ('raw', lambda s, e: RawVocabulary.extract_from_sentences(s, e)), 
    ('glove_fixed', lambda s, e: GloveVocabulary.extract_from_sentences(s, e, hold_fixed=True)),
    ('glove_not_fixed', lambda s, e: GloveVocabulary.extract_from_sentences(s, e, hold_fixed=False))
])

# Global options
OPTIONS = None

def _parse_args():
  global OPTIONS
  parser = argparse.ArgumentParser(
      description='Test neural alignment model on toy data.')
  parser.add_argument('--hidden-size', '-d', type=int,
                      help='Dimension of hidden units')
  parser.add_argument('--input-embedding-dim', '-i', type=int,
                      help='Dimension of input vectors.')
  parser.add_argument('--output-embedding-dim', '-o', type=int,
                      help='Dimension of output word vectors.')
  parser.add_argument('--num_epochs', '-t', type=int, default=0,
                      help='Number of epochs to train (default is no training).')
  parser.add_argument('--learning-rate', '-r', type=float, default=0.1,
                      help='Learning rate (default = 0.1).')
  parser.add_argument('--batch-size', '-b', type=int, default=1,
                      help='Size of mini-batch (default is SGD).')
  parser.add_argument('--continuous-spec', '-c',
                      help='type of continuous model (options: [%s])' % (
                          ', '.join(CONTINUOUS_MODELS)))
  parser.add_argument('--model', '-m',
                      help='type of overall model (options: [%s])' % (
                          ', '.join(MODELS)))
  parser.add_argument('--input-vocab-type',
                      help='type of input vocabulary (options: [%s])' % (
                          ', '.join(VOCAB_TYPES)), default='raw')
  parser.add_argument('--output-vocab-type',
                      help='type of output vocabulary (options: [%s])' % (
                          ', '.join(VOCAB_TYPES)), default='raw')
  parser.add_argument('--no-eos-on-output', action='store_true',
                      help='Do not add end of sentence token to output sequence.')
  parser.add_argument('--reverse-input', action='store_true',
                      help='Reverse the input sentence (intended for encoder-decoder).')
  parser.add_argument('--float32', action='store_true',
                      help='Use 32-bit floats (default is 64-bit/double precision).')
  parser.add_argument('--beam-size', '-k', type=int, default=0,
                      help='Use beam search with given beam size (default is greedy).')
  parser.add_argument('--train-data', help='Path to training data.')
  parser.add_argument('--dev-data', help='Path to dev data.')
  parser.add_argument('--save-file', help='Path to save parameters.')
  parser.add_argument('--load-file', help='Path to load parameters, will ignore other passed arguments.')
  parser.add_argument('--theano-fast-compile', action='store_true',
                      help='Run Theano in fast compile mode.')
  parser.add_argument('--theano-profile', action='store_true',
                      help='Turn on profiling in Theano.')
  parser.add_argument('--gpu', action='store_true', help='Use GPU.')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  OPTIONS = parser.parse_args()
  
  # Some basic error checking
  if OPTIONS.continuous_spec not in CONTINUOUS_MODELS:
    print >> sys.stderr, 'Error: continuous_spec must be in %s' % (
        ', '.join(CONTINUOUS_MODELS))
    sys.exit(1)
  if OPTIONS.model not in MODELS:
    print >> sys.stderr, 'Error: model must be in %s' % (
        ', '.join(MODELS))
    sys.exit(1)
  if OPTIONS.input_vocab_type not in VOCAB_TYPES:
    print >> sys.stderr, 'Error: input_vocab_type must be in %s' % (
        ', '.join(VOCAB_TYPES))
    sys.exit(1)
  if OPTIONS.output_vocab_type not in VOCAB_TYPES:
    print >> sys.stderr, 'Error: output_vocab_type must be in %s' % (
        ', '.join(VOCAB_TYPES))
    sys.exit(1)


def configure_theano():
  if OPTIONS.theano_fast_compile:
    theano.config.mode='FAST_COMPILE'
  else:
    theano.config.mode='FAST_RUN'
    theano.config.linker='cvm'
  if OPTIONS.theano_profile:
    theano.config.profile = True
  if OPTIONS.float32 or OPTIONS.gpu:
    theano.config.floatX = 'float32'

def load_dataset(filename):
  with open(filename) as f:
    return [tuple(line.rstrip('\n').split('\t')) for line in f]

def get_input_vocabulary(dataset):
  sentences = [x[0] for x in dataset]
  constructor = VOCAB_TYPES[OPTIONS.input_vocab_type]
  if OPTIONS.float32:
    return constructor(sentences, OPTIONS.input_embedding_dim,
                       float_type=numpy.float32)
  else:
    return constructor(sentences, OPTIONS.input_embedding_dim)

def get_output_vocabulary(dataset):
  sentences = [x[1] for x in dataset]
  constructor = VOCAB_TYPES[OPTIONS.output_vocab_type]
  if OPTIONS.float32:
    return constructor(sentences, OPTIONS.output_embedding_dim,
                       float_type=numpy.float32)
  else:
    return constructor(sentences, OPTIONS.output_embedding_dim)

def update_model(model, dataset):
  """Update model for new dataset if fixed word vectors were used."""
  need_new_model = False
  if OPTIONS.input_vocab_type == 'glove_fixed':
    in_vocabulary = get_input_vocabulary(dataset)
    need_new_model = True
  else:
    in_vocabulary = model.in_vocabulary

  if OPTIONS.output_vocab_type == 'glove_fixed':
    out_vocabulary = get_output_vocabulary(dataset)
    need_new_model = True
  else:
    out_vocabulary = model.out_vocabulary

  if need_new_model:
    spec = model.spec
    spec.set_in_vocabulary(in_vocabulary)
    spec.set_out_vocabulary(out_vocabulary)
    model = get_model(spec)  # Create a new model!
  return model

def preprocess_data(in_vocabulary, out_vocabulary, raw):
  eos_on_output = not OPTIONS.no_eos_on_output
  data = []
  for ex in raw:
    kwargs = {}
    if len(ex) == 2:
      x, y = ex
    else:
      x, y, actions_raw = ex
      actions = [int(a) for a in actions_raw.split(' ')]
      kwargs['actions'] = actions
    x_inds = in_vocabulary.sentence_to_indices(x)
    y_inds = out_vocabulary.sentence_to_indices(y, add_eos=eos_on_output)
    if OPTIONS.reverse_input:
      x_inds = x_inds[::-1]
    data.append((x_inds, y_inds, kwargs))
  return data

def get_continuous_spec(in_vocabulary, out_vocabulary):
  constructor = CONTINUOUS_MODELS[OPTIONS.continuous_spec]
  return constructor(in_vocabulary, out_vocabulary, OPTIONS.hidden_size)

def get_model(spec):
  constructor = MODELS[OPTIONS.model]
  if OPTIONS.float32:
    model = constructor(spec, float_type=numpy.float32)
  else:
    model = constructor(spec)
  return model

def print_accuracy_metrics(is_correct_list, tokens_correct_list,
                           x_len_list, y_len_list):
  # Overall metrics
  num_examples = len(is_correct_list)
  num_correct = sum(is_correct_list)
  num_tokens_correct = sum(tokens_correct_list)
  num_tokens = sum(y_len_list)
  seq_accuracy = float(num_correct) / num_examples
  token_accuracy = float(num_tokens_correct) / num_tokens

  # Per-length metrics
  num_correct_per_len = collections.defaultdict(int)
  tokens_correct_per_len = collections.defaultdict(int)
  num_per_len = collections.defaultdict(int)
  tokens_per_len = collections.defaultdict(int)
  for is_correct, tokens_correct, x_len, y_len in itertools.izip(
      is_correct_list, tokens_correct_list, x_len_list, y_len_list):
    num_correct_per_len[x_len] += is_correct
    tokens_correct_per_len[x_len] += tokens_correct
    num_per_len[x_len] += 1
    tokens_per_len[x_len] += y_len

  # Print sequence-level accuracy
  print 'Sequence-level accuracy: %d/%d = %g' % (num_correct, num_examples, seq_accuracy)
  for i in sorted(num_correct_per_len):
    cur_num_correct = num_correct_per_len[i]
    cur_num_examples = num_per_len[i]
    cur_accuracy = float(cur_num_correct) / cur_num_examples
    print '  input length = %d: %d/%d = %g correct' % (
        i - 1, cur_num_correct, cur_num_examples, cur_accuracy)

  # Print token-level accuracy
  print 'Token-level accuracy: %d/%d = %g' % (num_tokens_correct, num_tokens, token_accuracy)
  for i in sorted(tokens_correct_per_len):
    cur_num_tokens_correct = tokens_correct_per_len[i]
    cur_num_tokens = tokens_per_len[i]
    cur_accuracy = float(cur_num_tokens_correct) / cur_num_tokens
    print '  input length = %d: %d/%d = %g correct' % (
        i - 1, cur_num_tokens_correct, cur_num_tokens, cur_accuracy)

def decode(model, x_inds):
  if OPTIONS.beam_size == 0:
    return model.decode_greedy(x_inds, max_len=(2*len(x_inds)+50))
  else:
    return model.decode_beam(x_inds, beam_size=OPTIONS.beam_size)

def evaluate(model, in_vocabulary, out_vocabulary, dataset):
  """Evaluate the model.

  Supports dataset mapping x to multiple y.  If so, it treats
  any of those y as acceptable answers.
  """
  is_correct_list = []
  tokens_correct_list = []
  x_len_list = []
  y_len_list = []

  all_x_list = []
  all_x_set = set()
  x_to_inds = {}
  x_to_all_y = collections.defaultdict(list)
  x_to_kwargs = {}
  for x_inds, y_inds, kwargs in dataset:
    x_words = in_vocabulary.indices_to_sentence(x_inds)
    y_words = out_vocabulary.indices_to_sentence(y_inds)
    if x_words not in all_x_set:
      all_x_set.add(x_words)
      all_x_list.append(x_words)
    x_to_inds[x_words] = x_inds
    x_to_all_y[x_words].append((y_inds, y_words))
    x_to_kwargs[x_words] = kwargs

  for example_num, x_words in enumerate(all_x_list):
    x_inds = x_to_inds[x_words]
    y_all = x_to_all_y[x_words]
    y_inds_all = [y[0] for y in y_all]
    y_words_all = [y[1] for y in y_all]
    kwargs = x_to_kwargs[x_words]

    print 'Example %d' % example_num
    print '  x      = "%s"' % x_words
    print '  y      = "%s"' % y_words_all[0]

    y_pred = decode(model, x_inds)
    y_pred_words = out_vocabulary.indices_to_sentence(y_pred)

    # Compute accuracy metrics
    is_correct = (y_pred_words in y_words_all)
    if len(y_all) == 1:
      y_inds = y_inds_all[0]
      y_words = y_words_all[0]
      tokens_correct = sum(a == b for a, b in zip(y_pred, y_inds))
    else:
      tokens_correct = 0  # TODO(robinjia): Take the max over y_all?
    is_correct_list.append(is_correct)
    tokens_correct_list.append(tokens_correct)
    x_len_list.append(len(x_inds))
    y_len_list.append(len(y_inds))
    print '  y_pred = "%s"' % y_pred_words
    print '  sequence correct = %s' % is_correct
    print '  token accuracy = %d/%d = %g' % (
        tokens_correct, len(y_inds), float(tokens_correct) / len(y_inds))
  print_accuracy_metrics(is_correct_list, tokens_correct_list,
                         x_len_list, y_len_list)

def run():
  configure_theano()
  if OPTIONS.train_data:
    train_raw = load_dataset(OPTIONS.train_data)
  if OPTIONS.load_file:
    print >> sys.stderr, 'Loading saved params from %s' % OPTIONS.load_file
    spec = specutil.load(OPTIONS.load_file)
    in_vocabulary = spec.in_vocabulary
    out_vocabulary = spec.out_vocabulary
  elif OPTIONS.train_data:
    print >> sys.stderr, 'Initializing parameters...'
    in_vocabulary = get_input_vocabulary(train_raw)
    out_vocabulary = get_output_vocabulary(train_raw)
    spec = get_continuous_spec(in_vocabulary, out_vocabulary)
  else:
    raise Exception('Must either provide parameters to load or training data.')

  model = get_model(spec)

  if OPTIONS.train_data:
    train_data = preprocess_data(in_vocabulary, out_vocabulary, train_raw)
    model.train(train_data, T=OPTIONS.num_epochs, eta=OPTIONS.learning_rate,
                batch_size=OPTIONS.batch_size)

  if OPTIONS.save_file:
    print >> sys.stderr, 'Saving parameters...'
    spec.save(OPTIONS.save_file)

  if OPTIONS.train_data:
    print 'Training data:'
    evaluate(model, in_vocabulary, out_vocabulary, train_data)

  if OPTIONS.dev_data:
    dev_raw = load_dataset(OPTIONS.dev_data)
    dev_model = update_model(model, dev_raw)
    dev_data = preprocess_data(dev_model.in_vocabulary,
                               dev_model.out_vocabulary, dev_raw)
    print 'Testing data:'
    evaluate(dev_model, in_vocabulary, out_vocabulary, dev_data)

def main():
  _parse_args()
  run()

if __name__ == '__main__':
  main()
