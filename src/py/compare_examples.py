"""Used to compare examples contained in two files."""
import sys

def read(in_file):
  examples = []
  with open(in_file) as f:
    for line in f:
      x, y = line.split('\t') 
      examples.append((x.split(' '), y.split(' ')))
  return examples

def edit_distance(a, b):
  cache = {}
  def recurse(i, j):
    if (i, j) in cache:
      return cache[(i, j)]
    if i == 0: 
      ret = j
    elif j == 0: 
      ret = i
    else:
      ret = min(recurse(i-1, j) + 1,
                recurse(i, j-1) + 1,
                recurse(i-1, j-1) + int(a[i-1] != b[j-1]))
    cache[(i, j)] = ret
    return ret
  ans = recurse(len(a), len(b))
  return ans

def main(train_file, dev_file):
  train_examples = read(train_file)
  dev_examples = read(dev_file)
  for i, (x_dev, y_dev) in enumerate(dev_examples):
    scores = [(edit_distance(x_dev, ex[0]), ex)
               for ex in train_examples]
    scores.sort()
    print 'Dev Example %d' % i
    print '  x: %s' % ' '.join(x_dev)
    for j in range(5):
      print '  %d: %s (edit-distance %d)' % (
        j, ' '.join(scores[j][1][0]), scores[j][0])


if __name__ == '__main__':
  main(*sys.argv[1:])
