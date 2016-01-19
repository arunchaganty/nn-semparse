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
cl run :src :lib :evaluator :regex :regex-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention --stats-file stats.json --domain regex --train-data regex/regex_fold1of3_train440.tsv --dev-data regex/regex_fold1of3_test220.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n regex_fold1 -d 'no augmentation' $flags
sleep 1
cl run :src :lib :evaluator :regex :regex-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention --stats-file stats.json --domain regex --train-data regex-augmented/regex_train_fold1of3_train440_200str_200int.tsv --dev-data regex/regex_fold1of3_test220.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n regex_fold1 -d '200 str + 200 int' $flags
sleep 1
cl run :src :lib :evaluator :regex :regex-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention --stats-file stats.json --domain regex --train-data regex/regex_fold2of3_train440.tsv --dev-data regex/regex_fold2of3_test220.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n regex_fold2 -d 'no augmentation' $flags
sleep 1
cl run :src :lib :evaluator :regex :regex-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention --stats-file stats.json --domain regex --train-data regex-augmented/regex_train_fold2of3_train440_200str_200int.tsv --dev-data regex/regex_fold2of3_test220.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n regex_fold2 -d '200 str + 200 int' $flags
sleep 1
cl run :src :lib :evaluator :regex :regex-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention --stats-file stats.json --domain regex --train-data regex/regex_fold3of3_train440.tsv --dev-data regex/regex_fold3of3_test220.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n regex_fold3 -d 'no augmentation' $flags
sleep 1
cl run :src :lib :evaluator :regex :regex-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32,cuda.root=/usr/local/cuda-7.0 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention --stats-file stats.json --domain regex --train-data regex-augmented/regex_train_fold3of3_train440_200str_200int.tsv --dev-data regex/regex_fold3of3_test220.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n regex_fold3 -d '200 str + 200 int' $flags
