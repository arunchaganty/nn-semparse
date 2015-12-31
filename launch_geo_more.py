#!/usr/bin/env python
import subprocess
import os
import re
import sys
import time

if len(sys.argv) < 3:
  print >> sys.stderr, 'Usage: %s [hidden layer size] [word embedding size]' % sys.argv[0]
  sys.exit(1)

d = int(sys.argv[1])
i = int(sys.argv[2])
device = 'cpu'
num_cpu = 4

def launch(filename):
  n = int(re.search('train([0-9]+)', filename).group(1))
  m = re.search('augment([0-9]+k?)', filename)
  if m:
    description = '%d + augment %s' % (n, m.group(1))
  else:
    description = '%d only' % n
  vals = {'d': d, 'i': i, 'device': device, 'num_cpu': num_cpu, 'filename': filename, 'description': description}
  #print ' '.join([
  subprocess.call([
      'cl', 'run', ':src', ':lib', ':evaluator', ':geo880-lesscopy', ':geo880-augmented',
      'OMP_NUM_THREADS=%(num_cpu)d THEANO_FLAGS=blas.ldflags=-lopenblas,device=%(device)s,floatX=float32 python src/py/main.py -d %(d)d -i %(i)d -o %(i)d -p attention -u 1 -t 25,5,5 -c lstm -m attention --stats-file stats.json --domain geoquery --train-data %(filename)s --dev-data geo880-lesscopy/geo880_dev100.tsv --save-file params' % vals,
      '--request-queue', 'jag', '--request-cpus', str(num_cpu), '-n', 'geo_more', '-d', description])
  time.sleep(5)

filenames = [
    'geo880-lesscopy/geo880_train100.tsv',
    'geo880-augmented/geo880_train100_augment500.tsv',
    'geo880-augmented/geo880_train100_augment1k.tsv',

    'geo880-lesscopy/geo880_train200.tsv',
    'geo880-augmented/geo880_train200_augment500.tsv',
    'geo880-augmented/geo880_train200_augment1k.tsv',
#    'geo880-augmented/geo880_train200_augment2k.tsv',
#    'geo880-augmented/geo880_train200_augment3k.tsv',
#    'geo880-augmented/geo880_train200_augment4k.tsv',

    'geo880-lesscopy/geo880_train300.tsv',
    'geo880-augmented/geo880_train300_augment500.tsv',
    'geo880-augmented/geo880_train300_augment1k.tsv',
#    'geo880-augmented/geo880_train300_augment2k.tsv',
#    'geo880-augmented/geo880_train300_augment3k.tsv',
#    'geo880-augmented/geo880_train300_augment4k.tsv',
#    'geo880-augmented/geo880_train300_augment5k.tsv',

    'geo880-lesscopy/geo880_train400.tsv',
    'geo880-augmented/geo880_train400_augment500.tsv',
    'geo880-augmented/geo880_train400_augment1k.tsv',
#    'geo880-augmented/geo880_train400_augment2k.tsv',
#    'geo880-augmented/geo880_train400_augment3k.tsv',
#    'geo880-augmented/geo880_train400_augment4k.tsv',
#    'geo880-augmented/geo880_train400_augment5k.tsv',

    'geo880-lesscopy/geo880_train500.tsv',
    'geo880-augmented/geo880_train500_augment500.tsv',
    'geo880-augmented/geo880_train500_augment1k.tsv',
#    'geo880-augmented/geo880_train500_augment2k.tsv',
#    'geo880-augmented/geo880_train500_augment3k.tsv',
#    'geo880-augmented/geo880_train500_augment4k.tsv',
#    'geo880-augmented/geo880_train500_augment5k.tsv',
]

for fn in filenames:
  launch(fn)
