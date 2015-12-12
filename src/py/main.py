"""Run tests on toy data for IRW models."""
import argparse
import collections
import itertools
import json
import math
import numpy
import os
import random
import re
import subprocess
import sys
import tempfile
import theano

# Local imports
from encoderdecoder import EncoderDecoderModel
from attention import AttentionModel
from example import Example
from lexicon import Lexicon
import spec as specutil
from vocabulary import GloveVocabulary, RawVocabulary, Vocabulary

MODELS = collections.OrderedDict([
    ('encoderdecoder', EncoderDecoderModel),
    ('attention', AttentionModel),
])

VOCAB_TYPES = collections.OrderedDict([
    ('raw', lambda s, e, **kwargs: RawVocabulary.from_sentences(
        s, e, **kwargs)), 
    ('glove_fixed', lambda s, e, **kwargs: GloveVocabulary.from_sentences(
        s, e, hold_fixed=True, **kwargs)),
    ('glove_not_fixed', lambda s, e, **kwargs: GloveVocabulary.from_sentences(
        s, e, hold_fixed=False, **kwargs))
])

# Global options
OPTIONS = None

# Global statistics
STATS = {}

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
  parser.add_argument('--copy', '-p', default=None,
                      help='Way to copy words (options: [lexicon, attention, attention-logistic]).')
  parser.add_argument('--unk-cutoff', '-u', type=int, default=0,
                      help='Treat input words with <= this many occurrences as UNK.')
  parser.add_argument('--num_epochs', '-t', type=int, default=0,
                      help='Number of epochs to train (default is no training).')
  parser.add_argument('--learning-rate', '-r', type=float, default=0.1,
                      help='Learning rate (default = 0.1).')
  parser.add_argument('--batch-size', '-b', type=int, default=1,
                      help='Size of mini-batch (default is SGD).')
  parser.add_argument('--rnn-type', '-c',
                      help='type of continuous RNN model (options: [%s])' % (
                          ', '.join(specutil.RNN_TYPES)))
  parser.add_argument('--model', '-m',
                      help='type of overall model (options: [%s])' % (
                          ', '.join(MODELS)))
  parser.add_argument('--input-vocab-type',
                      help='type of input vocabulary (options: [%s])' % (
                          ', '.join(VOCAB_TYPES)), default='raw')
  parser.add_argument('--output-vocab-type',
                      help='type of output vocabulary (options: [%s])' % (
                          ', '.join(VOCAB_TYPES)), default='raw')
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
  parser.add_argument('--domain', help='Domain to evaluate (options: [geoquery,regex])')
  parser.add_argument('--stats-file', help='Path to save statistics (JSON format).')
  parser.add_argument('--shell', action='store_true', 
                      help='Start an interactive shell.')
  parser.add_argument('--server', action='store_true', 
                      help='Start an interactive web console (requires bottle).')
  parser.add_argument('--hostname', default='127.0.0.1', help='server hostname')
  parser.add_argument('--port', default=9001, type=int, help='server port')
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
  if OPTIONS.rnn_type not in specutil.RNN_TYPES:
    print >> sys.stderr, 'Error: rnn type must be in %s' % (
        ', '.join(specutil.RNN_TYPES))
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
                       unk_cutoff=OPTIONS.unk_cutoff,
                       float_type=numpy.float32)
  else:
    return constructor(sentences, OPTIONS.input_embedding_dim,
                       unk_cutoff=OPTIONS.unk_cutoff)

def get_output_vocabulary(dataset):
  sentences = [x[1] for x in dataset]
  constructor = VOCAB_TYPES[OPTIONS.output_vocab_type]
  if OPTIONS.float32:
    return constructor(sentences, OPTIONS.output_embedding_dim,
                       float_type=numpy.float32)
  else:
    return constructor(sentences, OPTIONS.output_embedding_dim)

def get_lexicon(dataset):
  if OPTIONS.copy == 'lexicon':
    sentences = [x[0] for x in dataset]
    # TODO(robinjia): load lexicon entries for test data too
    return Lexicon.from_sentences(sentences, OPTIONS.hidden_size,
                                  OPTIONS.unk_cutoff)
  else:
    return None

def update_model(model, dataset):
  """Update model for new dataset if fixed word vectors were used."""
  # Do the lexicon update in-place
  if model.lexicon:
    for x, y in dataset:
      words = x.split(' ')
      for w in words:
        model.lexicon.add_entry((w, w))

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

def preprocess_data(in_vocabulary, out_vocabulary, lexicon, raw):
  data = []
  for raw_ex in raw:
    x_str, y_str = raw_ex
    ex = Example(x_str, y_str, in_vocabulary, out_vocabulary, lexicon,
                 reverse_input=OPTIONS.reverse_input)
    data.append(ex)
  return data

def get_spec(in_vocabulary, out_vocabulary, lexicon):
  kwargs = {'rnn_type': OPTIONS.rnn_type}
  if OPTIONS.copy.startswith('attention'):
    if OPTIONS.model == 'attention':
      kwargs['attention_copying'] = OPTIONS.copy
    else:
      print >> sys.stderr, "Can't use use attention-based copying without attention model"
      sys.exit(1)
  constructor = MODELS[OPTIONS.model].get_spec_class()
  return constructor(in_vocabulary, out_vocabulary, lexicon, 
                     OPTIONS.hidden_size, **kwargs)

def get_model(spec):
  constructor = MODELS[OPTIONS.model]
  if OPTIONS.float32:
    model = constructor(spec, float_type=numpy.float32)
  else:
    model = constructor(spec)
  return model

def print_accuracy_metrics(name, is_correct_list, tokens_correct_list,
                           x_len_list, y_len_list, denotation_correct_list):
  # Overall metrics
  num_examples = len(is_correct_list)
  num_correct = sum(is_correct_list)
  num_tokens_correct = sum(tokens_correct_list)
  num_tokens = sum(y_len_list)
  seq_accuracy = float(num_correct) / num_examples
  token_accuracy = float(num_tokens_correct) / num_tokens

  STATS[name] = {}

  # Print sequence-level accuracy
  STATS[name]['sentence'] = {
      'correct': num_correct,
      'total': num_examples,
      'accuracy': seq_accuracy,
  }
  print 'Sequence-level accuracy: %d/%d = %g' % (num_correct, num_examples, seq_accuracy)

  # Print token-level accuracy
  STATS[name]['token'] = {
      'correct': num_tokens_correct,
      'total': num_tokens,
      'accuracy': token_accuracy,
  }
  print 'Token-level accuracy: %d/%d = %g' % (num_tokens_correct, num_tokens, token_accuracy)

  # Print denotation-level accuracy
  if denotation_correct_list:
    denotation_correct = sum(denotation_correct_list)
    denotation_accuracy = float(denotation_correct)/num_examples
    STATS[name]['denotation'] = {
        'correct': denotation_correct,
        'total': num_examples,
        'accuracy': denotation_accuracy
    }
    print 'Denotation-level accuracy: %d/%d = %g' % (denotation_correct, num_examples, denotation_accuracy)

def decode(model, ex):
  if OPTIONS.beam_size == 0:
    return model.decode_greedy(ex, max_len=100)
  else:
    return model.decode_beam(ex, beam_size=OPTIONS.beam_size)

def compare_answers_geoquery(true_answers, pred_answers):
  def format_lf(s):
    # Strip underscores, collapse spaces when not inside quotation marks
    toks = []
    in_quotes = False
    quoted_toks = []
    for t in s.split():
      if in_quotes:
        if t == "'":
          in_quotes = False
          toks.append('"%s"' % ' '.join(quoted_toks))
          quoted_toks = []
        else:
          quoted_toks.append(t)
      else:
        if t == "'":
          in_quotes = True
        else:
          if len(t) > 1 and t.startswith('_'):
            toks.append(t[1:])
          else:
            toks.append(t)
    lf = ''.join(toks)

    # Balance parentheses
    num_left_paren = sum(1 for c in lf if c == '(')
    num_right_paren = sum(1 for c in lf if c == ')')
    diff = num_left_paren - num_right_paren
    if diff > 0:
      lf = lf + ')' * diff
    return lf

  all_lfs = ([format_lf(s) for s in true_answers] +
             [format_lf(s) for s in pred_answers])
  tf_lines = ['_parse([query], %s).' % lf for lf in all_lfs]
  tf = tempfile.NamedTemporaryFile(suffix='.dlog')
  for line in tf_lines:
    print >>tf, line
    print line
  tf.flush()
  msg = subprocess.check_output(['evaluator/geoquery', tf.name])
  tf.close()

  def get_denotation(line):
    m = re.search('\{[^}]*\}', line)
    if m: 
      return m.group(0)
    else:
      return line.strip()
  denotations = [get_denotation(line)
                 for line in msg.split('\n')
                 if line.startswith('        Example')]

  def print_failures(dens, name):
    num_syntax_error = sum(d == 'Example FAILED TO PARSE' for d in dens)
    num_exec_error = sum(d == 'Example FAILED TO EXECUTE' for d in dens)
    print '%s: %d syntax errors, %d executor errors' % (
        name, num_syntax_error, num_exec_error)

  true_dens = denotations[:len(true_answers)]
  pred_dens = denotations[len(true_answers):]
  print_failures(true_dens, 'gold')
  print_failures(pred_dens, 'predicted')
  return [t == p for t, p in zip(true_dens, pred_dens)]

def compare_answers_regex(true_answers, pred_answers):
  def format_regex(r):
    return ''.join(r.split()).replace('_', ' ')
  ret = []

  for true_ans, pred_ans in zip(true_answers, pred_answers):
    msg = subprocess.check_output([
        'evaluator/regex', 
        '(%s)' % format_regex(true_ans), 
        '(%s)' % format_regex(pred_ans)])
    ret.append(msg.strip().endswith('true'))
  return ret

def compare_answers(true_answers, pred_answers):
  if OPTIONS.domain == 'geoquery':
    return compare_answers_geoquery(true_answers, pred_answers)
  elif OPTIONS.domain == 'regex':
    return compare_answers_regex(true_answers, pred_answers)
  else:
    raise ValueError('Unrecognized domain %s' % OPTIONS.domain)

def evaluate(name, model, in_vocabulary, out_vocabulary, lexicon, dataset):
  """Evaluate the model.

  TODO(robinjia): Support dataset mapping x to multiple y.  If so, it treats
  any of those y as acceptable answers.
  """
  is_correct_list = []
  tokens_correct_list = []
  x_len_list = []
  y_len_list = []

  preds = [decode(model, ex)[0] for ex in dataset]
  if OPTIONS.domain:
    true_answers = [ex.y_str for ex in dataset]
    pred_answers = [' '.join(p[1]) for p in preds]
    denotation_correct_list = compare_answers(true_answers, pred_answers)
  else:
    denotation_correct_list = None

  for i, ex in enumerate(dataset):
    print 'Example %d' % i
    print '  x      = "%s"' % ex.x_str
    print '  y      = "%s"' % ex.y_str
    prob, y_pred_toks = preds[i]
    y_pred_str = ' '.join(y_pred_toks)

    # Compute accuracy metrics
    is_correct = (y_pred_str == ex.y_str)
    tokens_correct = sum(a == b for a, b in zip(y_pred_toks, ex.y_toks))
    is_correct_list.append(is_correct)
    tokens_correct_list.append(tokens_correct)
    x_len_list.append(len(ex.x_toks))
    y_len_list.append(len(ex.y_toks))
    print '  y_pred = "%s"' % y_pred_str
    print '  sequence correct = %s' % is_correct
    print '  token accuracy = %d/%d = %g' % (
        tokens_correct, len(ex.y_toks), float(tokens_correct) / len(ex.y_toks))
    if denotation_correct_list:
      denotation_correct = denotation_correct_list[i]
      print '  denotation correct = %s' % denotation_correct
  print_accuracy_metrics(name, is_correct_list, tokens_correct_list,
                         x_len_list, y_len_list, denotation_correct_list)

def run_shell(model):
  print '==== Neural Network Semantic Parsing REPL ===='
  print ''
  print 'Enter an utterance:'
  while True:
    s = raw_input('> ').strip()
    example = Example(s, '', model.in_vocabulary, model.out_vocabulary,
                      model.lexicon, reverse_input=OPTIONS.reverse_input)
    print ''
    print 'Result:'
    preds = decode(model, example)
    for prob, y_toks in preds[:10]:
      y_str = ' '.join(y_toks)
      print '  [p=%f] %s' % (prob, y_str)
    print ''

def make_heatmap(x_str, y_str, attention_list, copy_list):
  """Make an HTML heatmap of attention."""
  def css_color(r, g, b):
    """r, g, b are in 0-1, make """
    r2 = int(r * 255)
    g2 = int(g * 255)
    b2 = int(b * 255)
    return 'rgb(%d,%d,%d)' % (r2, g2, b2)

  x_toks = x_str.split(' ') + ['EOS']
  if y_str == '':
    y_toks = ['EOS']
  else:
    y_toks = y_str.split(' ') + ['EOS']
  lines = ['<table>', '<tr>', '<td/>']
  for w in y_toks:
    lines.append('<td>%s</td>' % w)
  lines.append('</tr>')
  for i, w in enumerate(x_toks):
    lines.append('<tr>')
    lines.append('<td>%s</td>' % w)
    for j in range(len(y_toks)):
      do_copy = copy_list[j]
      if do_copy:
        color = css_color(1 - attention_list[j][i], 1 - attention_list[j][i], 1)
      else:
        color = css_color(1, 1 - attention_list[j][i], 1 - attention_list[j][i])
      lines.append('<td/ style="background-color: %s">' % color)
    lines.append('</tr>')
  lines.append('</table>')
  return '\n'.join(lines)

def run_server(model, hostname='127.0.0.1', port=9001):
  import bottle
  print '==== Neural Network Semantic Parsing Server ===='

  app = bottle.Bottle()
  
  @app.route('/debug')
  def debug():
    content = make_heatmap(
        'what states border texas',
        'answer ( A , ( state ( A ) , next_to ( A , B ) , const ( B , stateid ( texas ) ) ) )',
        [[0.0, 0.25, 0.5, 0.75, 1.0]] * 29)
    return bottle.template('main', prompt='Enter a new query', content=content)

  @app.route('/post_query')
  def post_query():
    query = bottle.request.params.get('query')
    print 'Received query: "%s"' % query
    example = Example(query, '', model.in_vocabulary, model.out_vocabulary,
                      model.lexicon, reverse_input=OPTIONS.reverse_input)
    preds = decode(model, example)
    lines = ['<b>Query: "%s"</b>' % query, '<ul>']
    for i, deriv in enumerate(preds[:10]):
      y_str = ' '.join(deriv.y_toks)
      lines.append('<li> %d. [p=%f] %s' % (i, deriv.p, y_str))
      lines.append(make_heatmap(query, y_str, deriv.attention_list, deriv.copy_list))
    lines.append('</ul>')

    content = '\n'.join(lines)
    return bottle.template('main', prompt='Enter a new query', content=content)

  @app.route('/')
  def index():
    return bottle.template('main', prompt='Enter a query', content='')

  bottle.run(app, host=hostname, port=port)

def run():
  configure_theano()
  if OPTIONS.train_data:
    train_raw = load_dataset(OPTIONS.train_data)
  if OPTIONS.load_file:
    print >> sys.stderr, 'Loading saved params from %s' % OPTIONS.load_file
    spec = specutil.load(OPTIONS.load_file)
    in_vocabulary = spec.in_vocabulary
    out_vocabulary = spec.out_vocabulary
    lexicon = spec.lexicon
  elif OPTIONS.train_data:
    print >> sys.stderr, 'Initializing parameters...'
    in_vocabulary = get_input_vocabulary(train_raw)
    out_vocabulary = get_output_vocabulary(train_raw)
    lexicon = get_lexicon(train_raw)
    spec = get_spec(in_vocabulary, out_vocabulary, lexicon)
  else:
    raise Exception('Must either provide parameters to load or training data.')

  model = get_model(spec)

  if OPTIONS.train_data:
    train_data = preprocess_data(in_vocabulary, out_vocabulary, lexicon, train_raw)
    model.train(train_data, T=OPTIONS.num_epochs, eta=OPTIONS.learning_rate,
                batch_size=OPTIONS.batch_size)

  if OPTIONS.save_file:
    print >> sys.stderr, 'Saving parameters...'
    spec.save(OPTIONS.save_file)

  if OPTIONS.train_data:
    print >> sys.stderr, 'Evaluating on training data...'
    print 'Training data:'
    evaluate('train', model, in_vocabulary, out_vocabulary, lexicon, train_data)

  if OPTIONS.dev_data:
    print >> sys.stderr, 'Evaluating on dev data...'
    dev_raw = load_dataset(OPTIONS.dev_data)
    dev_model = update_model(model, dev_raw)
    dev_data = preprocess_data(dev_model.in_vocabulary,
                               dev_model.out_vocabulary, 
                               dev_model.lexicon, dev_raw)
    print 'Dev data:'
    evaluate('dev', dev_model, in_vocabulary, out_vocabulary, lexicon, dev_data)

  if OPTIONS.stats_file:
    out = open(OPTIONS.stats_file, 'w')
    print >>out, json.dumps(STATS)
    out.close()

  if OPTIONS.shell:
    run_shell(model)
  elif OPTIONS.server:
    run_server(model, hostname=OPTIONS.hostname, port=OPTIONS.port)

def main():
  _parse_args()
  run()

if __name__ == '__main__':
  main()
