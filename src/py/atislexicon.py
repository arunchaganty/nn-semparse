"""Reads a lexicon for ATIS.

A lexicon simply maps natural language phrases to identifiers in the ATIS database.
"""
import collections
import itertools
import os
import re
import sys

DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/atis/db')

class Lexicon:
  """A Lexicon class.

  The lexicon stores two types of rules:
    1. Entries are pairs (name, entity), 
       where name could be a single word or multi-word phrase.
    2. Handlers are pairs (regex, func(match) -> entity).
       regex is checked to see if it matches any span of the input.
       If so, the function is applied to the match object to yield an entity.

  We additionally keep track of
    3. Unique words.  If a word |w| appears in exactly one entry (|n|, |e|),
       then a lower-precision rule maps |w| directly to |e|, even if the
       entire name |n| is not present.

  Rules take precedence in the order given: 1, then 2, then 3.
  Within each block, rules that match larger spans take precedence
  over ones that match shorter spans.
  """
  def __init__(self):
    self.entries = collections.OrderedDict()
    self.handlers = []
    self.unique_word_map = collections.OrderedDict()
    self.seen_words = set()

  def add_entries(self, entries):
    for name, entity in entries:
      # Update self.entries
      if name in self.entries:
        if self.entries[name] != entity:
          print >> sys.stderr, 'Collision detected: %s -> %s, %s' % (
              name, self.entries[name], entity)
          continue
      self.entries[name] = entity

      # Update self.unique_word_map
      for w in name.split():
        if w in self.seen_words:
          # This word is not unique!
          if w in self.unique_word_map:
            del self.unique_word_map[w]
        else:
          self.unique_word_map[w] = entity
          self.seen_words.add(w)

  def add_handler(self, regex, func):
    self.handlers.append((regex, func))

  def test_handlers(self, s):
    """Apply all handlers to a word; for debugging."""
    entities = []
    for regex, func in self.handlers:
      m = re.match(regex, s)
      if m:
        entities.append(func(m))
    print '  %s -> %s' % (s, entities)

  def map_over_sentence(self, words):
    """Apply unambiguous lexicon rules to an entire sentence.
    
    Args:
      words: A list of words
    Returns: A list of length len(words), where words[i] maps to retval[i]
    """
    entities = ['' for i in range(len(words))]
    ind_pairs = sorted(list(itertools.combinations(range(len(words) + 1), 2)),
                       key=lambda x: x[0] - x[1])

    # Entries
    for i, j in ind_pairs:
      if any(x for x in entities[i:j]): 
        # Something in this span has already been assinged
        continue
      span = ' '.join(words[i:j])
      if span in self.entries:
        entity = self.entries[span]
        for k in range(i, j):
          entities[k] = entity

    # Handlers
    for i, j in ind_pairs:
      if any(x for x in entities[i:j]): continue
      span = ' '.join(words[i:j])
      for regex, func in self.handlers:
        m = re.match(regex, span)
        if m:
          entity = func(m)
          for k in range(i, j):
            entities[k] = entity

    # Unique words
    for i in range(len(words)):
      if entities[i]: continue
      word = words[i]
      if word in self.unique_word_map:
        entities[i] = self.unique_word_map[word]

    return entities

def clean_id(s, id_suffix, strip=None):
  true_id = s.replace(' ', '_')
  if strip:
    for v in strip:
      true_id = true_id.replace(v, '').strip()
  return '%s:%s' % (true_id, id_suffix)

def clean_name(s, strip=None, split=None, prefix=None):
  if split:
    for v in split:
      s = s.replace(v, ' ')
  if strip:
    for v in strip:
      s = s.replace(v, '')
  #if s.endswith(', inc.') or s.endswith(', ltd.'): 
  #  s = s[:-6]
  #s = s.replace('/', ' ').replace('-', '').replace("'", '')
  if prefix:
    s = prefix + s
  return s

def read_db(basename, id_col, name_col, id_suffix,
            strip_id=None, strip_name=None, split_name=None, prefix_name=None):
  filename = os.path.join(DB_DIR, basename)
  data = []  # Pairs of (name, id)
  with open(filename) as f:
    for line in f:
      row = [s[1:-1] for s in re.findall('"[^"]*"|[0-9]+', line.strip())]
      cur_name = clean_name(row[name_col].lower(), strip=strip_name,
                            split=split_name, prefix=prefix_name)
      cur_id = clean_id(row[id_col].lower(), id_suffix, strip=strip_id)
      data.append((cur_name, cur_id))
  return data

def handle_times(lex):
  # Mod 12 deals with 12am/12pm special cases...
  lex.add_handler('([0-9]{1,2})am$',
                  lambda m: '%d00:_ti' % (int(m.group(1)) % 12))
  lex.add_handler('([0-9]{1,2})pm$',
                  lambda m: '%d00:_ti' % (int(m.group(1)) % 12 + 12))
  lex.add_handler('([0-9]{1,2})([0-9]{2})am$', 
                  lambda m: '%d%02d:_ti' % (int(m.group(1)) % 12, int(m.group(2))))
  lex.add_handler('([0-9]{1,2})([0-9]{2})pm$', 
                  lambda m: '%d%02d:_ti' % (int(m.group(1)) % 12 + 12, int(m.group(2))))

def handle_flight_numbers(lex):
  lex.add_handler('[0-9]{2,}$', lambda m: '%d:_fn' % int(m.group(0)))

def handle_dollars(lex):
  lex.add_handler('([0-9]+) dollars$', lambda m: '%d:_do' % int(m.group(1)))

def get_lexicon():
  DAYS_OF_WEEK = [
      (s, '%s:_da' % s) 
      for s in ('monday', 'tuesday', 'wednesday', 'thursday', 
                'friday', 'saturday', 'sunday')
  ]
  DATE_NUMBERS = [('%d' % i, '%d:_dn' % i) for i in range(1, 32)]
  MEALS = [(m, '%s:_me' % m) for m in ('breakfast', 'lunch', 'dinner', 'snack')]

  lex = Lexicon()
  lex.add_entries(read_db('CITY.TAB', 1, 1, '_ci', strip_id=['.']))
  lex.add_entries(DAYS_OF_WEEK)
  lex.add_entries(read_db('AIRLINE.TAB', 0, 1, '_al',
                          strip_name=[', inc.', ', ltd.']))
  handle_times(lex)
  lex.add_entries(read_db('INTERVAL.TAB', 0, 0, '_pd'))
  lex.add_entries(DAYS_OF_WEEK)
  lex.add_entries(read_db('MONTH.TAB', 1, 1, '_mn'))
  lex.add_entries(read_db('AIRPORT.TAB', 0, 1, '_ap',
                          strip_name=[], split_name=['/']))
  lex.add_entries(read_db('COMP_CLS.TAB', 1, 1, '_cl'))
  lex.add_entries(read_db('CLS_SVC.TAB', 0, 0, '_fb', prefix_name='code '))
  handle_flight_numbers(lex)
  lex.add_entries(MEALS)
  handle_dollars(lex)
  return lex

#   8207 _ci = cities
#    888 _da = days of the week
#    735 _al = airlines
#    607 _ti = times
#    594 _pd = time of day
#    404 _dn = date number
#    389 _mn = month
#    203 _ap = airport
#    193 _cl = class
#     72 _fb = fare code
#     65 _fn = flight number
#     52 _me = meal
#     50 _do = dollars
# ----------------------
#     28 _rc
#     23 _ac = aircraft
#     22 _yr
#      4 _mf
#      2 _dc
#      2 _st
#      1 _hr

def print_aligned(a, b, indent=0):
  a_toks = []
  b_toks = []
  for x, y in zip(a, b):
    cur_len = max(len(x), len(y))
    a_toks.append(x.ljust(cur_len))
    b_toks.append(y.ljust(cur_len))

  prefix = ' ' * indent
  print '%s%s' % (prefix, ' '.join(a_toks))
  print '%s%s' % (prefix, ' '.join(b_toks))

if __name__ == '__main__':
  # Print out the lexicon
  lex = get_lexicon()
  print 'Lexicon entries:'
  for name, entity in lex.entries.iteritems():
    print '  %s -> %s' % (name, entity)
  print 'Unique word map:'
  for word, entity in lex.unique_word_map.iteritems():
    print '  %s -> %s'  % (word, entity)

  print 'Test cases:'
  lex.test_handlers('8am')
  lex.test_handlers('8pm')
  lex.test_handlers('12am')
  lex.test_handlers('12pm')
  lex.test_handlers('832am')
  lex.test_handlers('904am')
  lex.test_handlers('1204am')
  lex.test_handlers('832pm')
  lex.test_handlers('904pm')
  lex.test_handlers('1204pm')
  lex.test_handlers('21')
  lex.test_handlers('4341')
  lex.test_handlers('4341 dollars')

  with open('data/atis/processed/atis_train.tsv') as f:
    for line in f:
      words = line.split('\t')[0].split(' ')
      entities = lex.map_over_sentence(words) 
      print '-' * 80
      print_aligned(words, entities, indent=2)
