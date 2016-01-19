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
num_cpu = 4

def launch(filename):
  # Filenames are like regex_train_sm_100str_100int.tsv
  basename = os.path.basename(filename)
  cl_filename = os.path.join('regex-augmented', basename)
  description = basename[15:-4]
  vals = {'d': d, 'i': i, 'device': device, 'num_cpu': num_cpu, 'filename': cl_filename, 'description': description}
  #print ' '.join([
  subprocess.call([
      'cl', 'run', ':src', ':lib', ':evaluator', ':regex', ':regex-augmented',
      'OMP_NUM_THREADS=%(num_cpu)d THEANO_FLAGS=blas.ldflags=-lopenblas,device=%(device)s,floatX=float32 python src/py/main.py -d %(d)d -i %(i)d -o %(i)d -p attention -u 1 -t 25,5,5 -c lstm -m attention --stats-file stats.json --domain regex --train-data %(filename)s --dev-data regex/regex_dev.tsv --save-file params' % vals,
      '--request-queue', 'jag', '--request-cpus', str(num_cpu), '-n', 'regex_tune', '-d', description])
  time.sleep(1)

filenames = sorted(glob.glob('data/regex/processed-augmented/*.tsv'))

for fn in filenames:
  launch(fn)
