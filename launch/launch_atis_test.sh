#!/usr/bin/env bash
if [ -z $2 ]
then
  echo "Usage: $0 [hidden layer size] [word embedding size]" 1>&2
  exit 1
fi
d=$1
i=$2 
device=cpu
queue=jag
num_cpu=8
cl run :src :lib :evaluator :atis :atis-db "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=cpu,floatX=float32 python src/py/main.py -d $d -i $i -o $i -u 1 -t 25,5,5 -c lstm -m encoderdecoder --stats-file stats.json -l --domain atis --dev-seed 0 --model-seed 0 --train-data atis/atis_train.tsv --dev-data atis/atis_test.tsv --save-file params" --request-docker-image robinjia/robinjia-codalab --request-queue jag --request-cpus $num_cpu -n atis_test -d "encDec, no aug"
sleep 1
cl run :src :lib :evaluator :atis :atis-db "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=cpu,floatX=float32 python src/py/main.py -d $d -i $i -o $i -u 1 -t 25,5,5 -c lstm -m encoderdecoder --stats-file stats.json -l --domain atis -a double:2000 --dev-seed 0 --model-seed 0 --train-data atis/atis_train.tsv --dev-data atis/atis_test.tsv --save-file params" --request-docker-image robinjia/robinjia-codalab --request-queue jag --request-cpus $num_cpu -n atis_test -d "encDec, double:2000"
sleep 1
cl run :src :lib :evaluator :atis :atis-db "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=cpu,floatX=float32 python src/py/main.py -d $d -i $i -o $i -u 1 -t 25,5,5 -c lstm -m attention --stats-file stats.json -l --domain atis --dev-seed 0 --model-seed 0 --train-data atis/atis_train.tsv --dev-data atis/atis_test.tsv --save-file params" --request-docker-image robinjia/robinjia-codalab --request-queue jag --request-cpus $num_cpu -n atis_test -d "attnBaseline, no aug"
sleep 1
cl run :src :lib :evaluator :atis :atis-db "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=cpu,floatX=float32 python src/py/main.py -d $d -i $i -o $i -u 1 -t 25,5,5 -c lstm -m attention --stats-file stats.json -l --domain atis -a double:2000 --dev-seed 0 --model-seed 0 --train-data atis/atis_train.tsv --dev-data atis/atis_test.tsv --save-file params" --request-docker-image robinjia/robinjia-codalab --request-queue jag --request-cpus $num_cpu -n atis_test -d "attnBaseline, double:2000"
sleep 1
cl run :src :lib :evaluator :atis :atis-db "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=cpu,floatX=float32 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention --stats-file stats.json -l --domain atis --dev-seed 0 --model-seed 0 --train-data atis/atis_train.tsv --dev-data atis/atis_test.tsv --save-file params" --request-docker-image robinjia/robinjia-codalab --request-queue jag --request-cpus $num_cpu -n atis_test -d "full, no aug"
sleep 1
cl run :src :lib :evaluator :atis :atis-db "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=cpu,floatX=float32 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention --stats-file stats.json -l --domain atis -a double:2000 --dev-seed 0 --model-seed 0 --train-data atis/atis_train.tsv --dev-data atis/atis_test.tsv --save-file params" --request-docker-image robinjia/robinjia-codalab --request-queue jag --request-cpus $num_cpu -n atis_test -d "full, double:2000"
sleep 1
