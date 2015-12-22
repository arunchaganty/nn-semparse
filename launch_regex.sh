#!/usr/bin/env bash
if [ -z $2 ]
then
  echo "Usage: $0 [hidden layer size] [word embedding size]" 1>&2
  exit 1
fi
d=$1
i=$2 
device=cpu
num_cpu=4
#flags='--request-docker-image codalab/theano-cuda7.0-352.39'
cl run :src :lib :evaluator :regex :regex-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention --stats-file stats.json --domain regex --train-data regex/regex_train_sm.tsv --dev-data regex/regex_dev.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n regex_tune -d 'no augmentation' $flags
sleep 5
cl run :src :lib :evaluator :regex :regex-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention --stats-file stats.json --domain regex --train-data regex-augmented/regex_train_sm_augmentInt.tsv --dev-data regex/regex_dev.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n regex_tune -d 'augment int' $flags
sleep 5
cl run :src :lib :evaluator :regex :regex-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention --stats-file stats.json --domain regex --train-data regex-augmented/regex_train_sm_augment1kStr.tsv --dev-data regex/regex_dev.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n regex_tune -d 'augment 1k str' $flags
sleep 5
cl run :src :lib :evaluator :regex :regex-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention --stats-file stats.json --domain regex --train-data regex-augmented/regex_train_sm_augment1kStrPlusInt.tsv --dev-data regex/regex_dev.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n regex_tune -d 'augment 1k str + int' $flags
