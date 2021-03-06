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
import atislexicon
import augmentation
from encoderdecoder import EncoderDecoderModel
from attention import AttentionModel
from example import Example
import spec as specutil
from vocabulary import Vocabulary

MODELS = collections.OrderedDict([
    ('encoderdecoder', EncoderDecoderModel),
    ('attention', AttentionModel),
])

VOCAB_TYPES = collections.OrderedDict([
    ('raw', lambda s, e, **kwargs: Vocabulary.from_sentences(
        s, e, **kwargs)), 
    ('glove', lambda s, e, **kwargs: Vocabulary.from_sentences(
        s, e, use_glove=True, **kwargs))
])

# Global options
OPTIONS = None

# Global statistics
STATS = {}

def _parse_args():
  global OPTIONS
  parser = argparse.ArgumentParser(
      description='A neural semantic parser.',
      formatter_class=argparse.RawTextHelpFormatter
  )
  parser.add_argument('--hidden-size', '-d', type=int,
                      help='Dimension of hidden units')
  parser.add_argument('--input-embedding-dim', '-i', type=int,
                      help='Dimension of input vectors.')
  parser.add_argument('--output-embedding-dim', '-o', type=int,
                      help='Dimension of output word vectors.')
  parser.add_argument('--copy', '-p', default='none',
                      help='Way to copy words (options: [none, attention, attention-logistic]).')
  parser.add_argument('--unk-cutoff', '-u', type=int, default=0,
                      help='Treat input words with <= this many occurrences as UNK.')
  parser.add_argument('--num-epochs', '-t', default=[],
                      type=lambda s: [int(x) for x in s.split(',')], 
                      help=('Number of epochs to train (default is no training).'
                            'If comma-separated list, will run for some epochs, halve learning rate, etc.'))
  parser.add_argument('--learning-rate', '-r', type=float, default=0.1,
                      help='Initial learning rate (default = 0.1).')
  parser.add_argument('--step-rule', '-s', default='simple',
                      help='Use a special SGD step size rule (types=[simple, adagrad, rmsprop,nesterov])')
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
  parser.add_argument('--domain', default=None,
                      help='Domain for augmentation and evaluation (options: [geoquery,regex,atis])')
  parser.add_argument('-l', '--lexicon', action='store_true',
                      help='Use a lexicon for copying (should also supply --domain)')
  parser.add_argument('--augment', '-a',
                      help=('Options for augmentation.  Expects list of key-int pairs, '
                            'e.g. "int:100,str:200".  Meaning depends on domain.'))
  parser.add_argument('--train-data', help='Path to training data.')
  parser.add_argument('--dev-data', help='Path to dev data.')
  parser.add_argument('--dev-frac', type=float, default=0.0,
                      help='Take this fraction of train data as dev data.')
  parser.add_argument('--dev-seed', type=int, default=0,
                      help='RNG seed for the train/dev splits (default = 0)')
  parser.add_argument('--model-seed', type=int, default=0,
                      help="RNG seed for the model's initialization and SGD ordering (default = 0)")
  parser.add_argument('--no-train', action='store_true', help="Avoid training")
  parser.add_argument('--save-file', default='out.params', help='Path to save parameters.')
  parser.add_argument('--load-file', help='Path to load parameters, will ignore other passed arguments.')
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
  parser.add_argument('--input', type=argparse.FileType('r'), 
                        default=sys.stdin,
                      help='File to read as input')
  parser.add_argument('--output', type=argparse.FileType('w'), 
                        default=sys.stdout,
                      help='File to read as input')

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

def update_model(model, dataset):
  """Update model for new dataset if fixed word vectors were used.
  
  Note: glove_fixed has been removed for now.
  """
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

def preprocess_data(model, raw):
  in_vocabulary = model.in_vocabulary
  out_vocabulary = model.out_vocabulary
  lexicon = model.lexicon

  data = []
  for raw_ex in raw:
    x_str, y_str = raw_ex
    ex = Example(x_str, y_str, in_vocabulary, out_vocabulary, lexicon,
                 reverse_input=OPTIONS.reverse_input)
    data.append(ex)
  return data

def get_spec(in_vocabulary, out_vocabulary, lexicon):
  kwargs = {'rnn_type': OPTIONS.rnn_type, 'step_rule': OPTIONS.step_rule}
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

def geo_format_lf(s):
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

def geo_get_denotation(line):
  m = re.search('\{[^}]*\}', line)
  if m: 
    return m.group(0)
  else:
    return line.strip()

def geo_print_failures(dens, name):
  num_syntax_error = sum(d == 'Example FAILED TO PARSE' for d in dens)
  num_exec_error = sum(d == 'Example FAILED TO EXECUTE' for d in dens)
  num_join_error = sum('Join failed syntactically' in d for d in dens)
  print '%s: %d syntax errors, %d executor errors' % (
      name, num_syntax_error, num_exec_error)

def geo_is_error(d):
  return 'FAILED' in d or 'Join failed syntactically' in d

def compare_answers_geoquery(true_answers, all_derivs):
  all_lfs = ([geo_format_lf(s) for s in true_answers] +
             [geo_format_lf(' '.join(d.y_toks)) 
              for x in all_derivs for d in x])
  tf_lines = ['_parse([query], %s).' % lf for lf in all_lfs]
  tf = tempfile.NamedTemporaryFile(suffix='.dlog')
  for line in tf_lines:
    print >>tf, line
    print line
  tf.flush()
  msg = subprocess.check_output(['evaluator/geoquery', tf.name])
  tf.close()
  denotations = [geo_get_denotation(line)
                 for line in msg.split('\n')
                 if line.startswith('        Example')]
  true_dens = denotations[:len(true_answers)]
  all_pred_dens = denotations[len(true_answers):]

  # Find the top-scoring derivation that executed without error
  derivs = []
  pred_dens = []
  cur_start = 0
  for deriv_set in all_derivs:
    for i in range(len(deriv_set)):
      cur_denotation = all_pred_dens[cur_start + i]
      if not geo_is_error(cur_denotation):
        derivs.append(deriv_set[i])
        pred_dens.append(cur_denotation)
        break
    else:
      derivs.append(deriv_set[0])  # Default to first derivation
      pred_dens.append(all_pred_dens[cur_start])
    cur_start += len(deriv_set)

  geo_print_failures(true_dens, 'gold')
  geo_print_failures(pred_dens, 'predicted')
  for t, p in zip(true_dens, pred_dens):
    print '%s: %s == %s' % (t == p, t, p)
  return derivs, [t == p for t, p in zip(true_dens, pred_dens)]

def compare_answers_regex(true_answers, all_derivs):
  def format_regex(r):
    r = ''.join(r.split()).replace('_', ' ')
    # Balance parenthese
    num_left_paren = sum(1 for c in r if c == '(')
    num_right_paren = sum(1 for c in r if c == ')')
    diff = num_left_paren - num_right_paren
    if diff > 0:
      r = r + ')' * diff
    elif diff < 0:
      r = ('(' * (-diff)) + r
    return r
  derivs = []
  is_correct_list = []
  for true_ans, deriv_list in zip(true_answers, all_derivs):
    if true_ans == ' '.join(deriv_list[0].y_toks):
      derivs.append(deriv_list[0])
      is_correct_list.append(True)
      continue
    for deriv in deriv_list:
      pred_ans = ' '.join(deriv.y_toks)
      msg = subprocess.check_output([
          'evaluator/regex', 
          '(%s)' % format_regex(true_ans), 
          '(%s)' % format_regex(pred_ans)])
      if msg.startswith('Error converting'): continue
      derivs.append(deriv)
      is_correct_list.append(msg.strip().endswith('true'))
      break
    else:
      derivs.append(deriv_list[0])  # Default to first derivation
      is_correct_list.append(False)
  return derivs, is_correct_list

def atis_standardize_varnames(ans):
  toks = ans.split(' ')
  varnames = {}
  new_toks = []
  for t in toks:
    if t == 'x' or t.startswith('$'):
      # t is a variable name
      if t in varnames:
        new_toks.append(varnames[t])
      else:
        new_varname = '$v%d' % len(varnames)
        varnames[t] = new_varname
        new_toks.append(new_varname)
    else:
      new_toks.append(t)
  lf = ' '.join(new_toks)
  return lf

def to_lisp_tree(expr):
  toks = expr.split(' ')
  def recurse(i):
    if toks[i] == '(':
      subtrees = []
      j = i+1
      while True:
        subtree, j = recurse(j)
        subtrees.append(subtree)
        if toks[j] == ')':
          return subtrees, j + 1
    else:
      return toks[i], i+1

  try:
    lisp_tree, final_ind = recurse(0)
    return lisp_tree
  except Exception as e:
    print >> sys.stderr, 'Failed to convert "%s" to lisp tree' % expr
    print e
    return None

def atis_sort_args(ans):
  lisp_tree = to_lisp_tree(ans)
  if lisp_tree is None: return ans

  # Post-order traversal, sorting as we go
  def recurse(node):
    if isinstance(node, str): return
    for child in node:
      recurse(child)
    if node[0] in ('_and', '_or'):
      node[1:] = sorted(node[1:], key=lambda x: str(x))
  recurse(lisp_tree)

  def tree_to_str(node):
    if isinstance(node, str):
      return node
    else:
      return '( %s )' % ' '.join(tree_to_str(child) for child in node)
  return tree_to_str(lisp_tree)

def atis_normalize(ans):
  """Perform some normalization of ATIS logical forms

  1. Standardize variable names, since atis_test uses different convention.
  2. Sort arguments to commutative operators such as _and() and _or().
  3. Balance parentheses.
  """
  lf = atis_standardize_varnames(ans)
  # Balance parentheses
  # In particular, atis_test gold lf's have unbalanced parens...
  num_left_paren = sum(1 for c in lf if c == '(')
  num_right_paren = sum(1 for c in lf if c == ')')
  diff = num_left_paren - num_right_paren
  if diff > 0:
    lf = lf + ' )' * diff
  lf = atis_sort_args(lf)
  return lf

def compare_answers_atis(true_answers, all_derivs):
  derivs = [x[0] for x in all_derivs]
  pred_answers = [' '.join(d.y_toks) for d in derivs]
  is_correct_list = []
  for true_ans, pred_ans in zip(true_answers, pred_answers):
    is_correct_list.append(
        atis_normalize(true_ans) == atis_normalize(pred_ans))
  return derivs, is_correct_list

def compare_answers(true_answers, all_derivs):
  if OPTIONS.domain == 'geoquery':
    return compare_answers_geoquery(true_answers, all_derivs)
  elif OPTIONS.domain == 'regex':
    return compare_answers_regex(true_answers, all_derivs)
  elif OPTIONS.domain == 'atis':
    return compare_answers_atis(true_answers, all_derivs)
  else:
    raise ValueError('Unrecognized domain %s' % OPTIONS.domain)

def evaluate(name, model, dataset):
  """Evaluate the model.

  TODO(robinjia): Support dataset mapping x to multiple y.  If so, it treats
  any of those y as acceptable answers.
  """
  in_vocabulary = model.in_vocabulary
  out_vocabulary = model.out_vocabulary

  is_correct_list = []
  tokens_correct_list = []
  x_len_list = []
  y_len_list = []

  if OPTIONS.domain:
    all_derivs = [decode(model, ex) for ex in dataset]
    true_answers = [ex.y_str for ex in dataset]
    derivs, denotation_correct_list = compare_answers(true_answers, all_derivs)
  else:
    derivs = [decode(model, ex)[0] for ex in dataset]
    denotation_correct_list = None

  for i, ex in enumerate(dataset):
    print 'Example %d' % i
    print '  x      = "%s"' % ex.x_str
    print '  y      = "%s"' % ex.y_str
    prob = derivs[i].p
    y_pred_toks = derivs[i].y_toks
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
    for pred in preds[:10]:
      prob, y_toks = pred.p, pred.y_toks
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

def load_raw_all():
  # Load train, and dev too if dev-frac was provided
  random.seed(OPTIONS.dev_seed)
  if OPTIONS.train_data:
    train_raw = load_dataset(OPTIONS.train_data)
    if OPTIONS.dev_frac > 0.0:
      num_dev = int(round(len(train_raw) * OPTIONS.dev_frac))
      random.shuffle(train_raw)
      dev_raw = train_raw[:num_dev]
      train_raw = train_raw[num_dev:]
      print >> sys.stderr, 'Split dataset into %d train, %d dev examples' % (
          len(train_raw), len(dev_raw))
    else:
      dev_raw = None
  else:
    train_raw = None
    dev_raw = None

  # Load dev data from separate file
  if OPTIONS.dev_data:
    if dev_raw:
      raise ValueError('dev-data and dev-frac conflict')
    dev_raw = load_dataset(OPTIONS.dev_data)

  # Perform augmentation if requested
  if train_raw and OPTIONS.augment:
    augmenter = augmentation.new(OPTIONS.domain, train_raw)
    aug_requests = [x.split(':') for x in OPTIONS.augment.split(',')]
    aug_requests = [(x[0], int(x[1])) for x in aug_requests]
    all_data = [train_raw]
    for name, num in aug_requests:
      all_data.append(augmenter.augment(name, num))
    train_raw = [ex for d in all_data for ex in d]

  return train_raw, dev_raw

def get_lexicon():
  if OPTIONS.lexicon:
    if OPTIONS.domain == 'atis':
      return atislexicon.get_lexicon()
    raise Exception('No lexicon for domain %s' % OPTIONS.domain)
  return None

def init_spec(train_raw):
  if OPTIONS.load_file:
    print >> sys.stderr, 'Loading saved params from %s' % OPTIONS.load_file
    spec = specutil.load(OPTIONS.load_file)
  elif OPTIONS.train_data:
    print >> sys.stderr, 'Initializing parameters...'
    in_vocabulary = get_input_vocabulary(train_raw)
    out_vocabulary = get_output_vocabulary(train_raw)
    lexicon = get_lexicon()
    spec = get_spec(in_vocabulary, out_vocabulary, lexicon)
  else:
    raise Exception('Must either provide parameters to load or training data.')
  return spec

def evaluate_train(model, train_data):
  print >> sys.stderr, 'Evaluating on training data...'
  print 'Training data:'
  evaluate('train', model, train_data)

def evaluate_dev(model, dev_raw):
  print >> sys.stderr, 'Evaluating on dev data...'
  dev_model = update_model(model, dev_raw)
  dev_data = preprocess_data(dev_model, dev_raw)
  print 'Dev data:'
  evaluate('dev', dev_model, dev_data)

def write_stats():
  if OPTIONS.stats_file:
    out = open(OPTIONS.stats_file, 'w')
    print >>out, json.dumps(STATS)
    out.close()

def run():
  configure_theano()
  train_raw, dev_raw = load_raw_all()
  random.seed(OPTIONS.model_seed)
  numpy.random.seed(OPTIONS.model_seed)
  spec = init_spec(train_raw)
  model = get_model(spec)

  if train_raw:
    train_data = preprocess_data(model, train_raw)
    random.seed(OPTIONS.model_seed)
    model.train(train_data, T=OPTIONS.num_epochs, eta=OPTIONS.learning_rate)

    if OPTIONS.save_file:
      print >> sys.stderr, 'Saving parameters...'
      spec.save(OPTIONS.save_file)
      model.save(OPTIONS.save_model_file)

  if train_raw:
    evaluate_train(model, train_data)
  if dev_raw:
    evaluate_dev(model, dev_raw)

  write_stats()

  if OPTIONS.shell:
    run_shell(model)
  elif OPTIONS.server:
    run_server(model, hostname=OPTIONS.hostname, port=OPTIONS.port)

def run_eval():
  import csv
  assert OPTIONS.load_file is not None
  assert OPTIONS.input is not None
  train_raw = load_dataset(OPTIONS.train_data)
  random.seed(OPTIONS.model_seed)
  numpy.random.seed(OPTIONS.model_seed)
  spec = init_spec(train_raw)
  model = get_model(spec)

  reader = csv.reader(OPTIONS.input, delimiter='\t')
  writer = csv.writer(OPTIONS.output, delimiter='\t')
  header = next(reader)
  #assert header == ['id', 'input']
  writer.writerow(['id','input', 'output', 'score'])
  for id, input in reader:
      s = input.strip()
      example = Example(s, '', model.in_vocabulary, model.out_vocabulary,
                          model.lexicon, reverse_input=OPTIONS.reverse_input)

      deriv = decode(model, example)[0]
      output = " ".join(deriv.y_toks).strip()
      score = deriv.p
      writer.writerow([id, input, output, score])

def main():
  _parse_args()
  if OPTIONS.no_train:
      run_eval()
  else:
      run()

if __name__ == '__main__':
  main()
