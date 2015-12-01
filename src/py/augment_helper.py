"""Some data augmentation schemes."""
import argparse
import collections
import sys

# Global options
OPTIONS = None

def _parse_args():
  global OPTIONS
  parser = argparse.ArgumentParser(description='Helper to augment data.')
  parser.add_argument('data_file', metavar='data.tsv',
                      help='Data file to augment.')
  parser.add_argument('--min-count', '-c', type=int, default=2,
                      help='Minimum number of occurrences of lexicon item')
  parser.add_argument('--max-len', '-l', type=int, default=6,
                      help='Maximum length (number of words) of lexicon item')
  parser.add_argument('--min-len', '-m', type=int, default=0,
                      help='Minimum length (number of words) of lexicon item')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  OPTIONS = parser.parse_args()

def read(in_file):
  examples = []
  with open(in_file) as f:
    for line in f:
      x, y = line.split('\t') 
      examples.append((x.split(' '), y.split(' ')))
  return examples

def make_ngrams(elems, min_len=None, max_len=None):
  ngrams = []
  for j in range(len(elems)):
    for i in range(j):
      if max_len and j - i > max_len: continue
      if min_len and j - i < min_len: continue
      ngrams.append(' '.join(elems[i:j]))
  return ngrams

def make_x_ngrams(elems):
  return make_ngrams(elems, min_len=OPTIONS.min_len, max_len=OPTIONS.max_len)

def make_y_ngrams(elems):
  def normalize(y):
    new_y = []
    old_to_new = {}
    next_var = 'A'
    for w in y.split(' '):
      if len(w) == 1 and ord(w[0]) >= ord('A') and ord(w[0]) <= ord('Z'):
        if w in old_to_new:
          new_y.append(old_to_new[w])
        else:
          old_to_new[w] = next_var
          new_y.append(next_var)
          next_var = chr(ord(next_var) + 1)
      else:
        new_y.append(w)
    return ' '.join(new_y)

  def is_balanced(y):
    num_paren = 0
    for w in y.split(' '):
      if w == '(':
        num_paren += 1
      elif w == ')':
        num_paren -= 1
      if num_paren < 0: return False
    return num_paren == 0

  return [normalize(y) for y in make_ngrams(elems) if is_balanced(y)]

def count(lists):
  ret = collections.defaultdict(int)
  for l in lists:
    for elem in l:
      ret[elem] += 1
  return ret

def make_x_to_y(data, x_counts):
  # Map x ngrams to y ngrams that show up in every example of that x ngram
  ret = {}
  for ex in data:
    x_ngrams = make_x_ngrams(ex[0])
    y_ngrams = set(make_y_ngrams(ex[1]))
    for x in x_ngrams:
      if x_counts[x] < OPTIONS.min_count: continue
      if x in ret:
        ret[x] = ret[x] & y_ngrams
      else:
        ret[x] = y_ngrams
  return ret

def make_lexicon(x_to_y_ngrams, y_ngram_counts):
  lexicon = {}
  for x in x_to_y_ngrams:
    y_ngrams = x_to_y_ngrams[x]

    # Choose the rarest one
    best = min((y_ngram_counts[y], y) for y in y_ngrams)[1]
    lexicon[x] = best
  return lexicon

def main():
  data = read(OPTIONS.data_file)
  x_ngrams = [make_x_ngrams(x) for x, y in data]
  y_ngrams = [make_y_ngrams(y) for x, y in data]
  x_ngram_counts = count(x_ngrams)
  y_ngram_counts = count(y_ngrams)
  x_to_y_ngrams = make_x_to_y(data, x_ngram_counts)
  lexicon = make_lexicon(x_to_y_ngrams, y_ngram_counts)
  for x in lexicon:
    print '%s: %s' % (x, lexicon[x])
  
if __name__ == '__main__':
  _parse_args()
  main()
