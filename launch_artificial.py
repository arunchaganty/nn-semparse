#!/usr/bin/env python
import glob
import os
import re
import subprocess
import sys
import time

if len(sys.argv) < 3:
  print >> sys.stderr, 'Usage: %s [hidden layer size] [word embedding size]' % sys.argv[0]
  sys.exit(1)

d = int(sys.argv[1])
i = int(sys.argv[2])
device = 'cpu'
num_cpu = 2

def launch(filename):
  # Filenames are like train_nested150.tsv
  basename = os.path.basename(filename)
  cl_filename = os.path.join('artificial', basename)
  description = basename[6:-4]
  vals = {'d': d, 'i': i, 'device': device, 'num_cpu': num_cpu, 'filename': cl_filename, 'description': description}
  #print ' '.join([
  subprocess.call([
      'cl', 'run', ':src', ':artificial',
      'OMP_NUM_THREADS=%(num_cpu)d THEANO_FLAGS=blas.ldflags=-lopenblas,device=%(device)s,floatX=float32 python src/py/main.py -d %(d)d -i %(i)d -o %(i)d -p attention -u 0 -t 30 -c lstm -m attention --stats-file stats.json --train-data %(filename)s --dev-data artificial/nested_test500.tsv --save-file params' % vals,
      '--request-queue', 'jag', '--request-cpus', str(num_cpu), '-n', 'artificial_train', '-d', description])
  time.sleep(1)

filenames = sorted(glob.glob('data/artificial/version01/train_*.tsv'))

for fn in filenames:
  launch(fn)
