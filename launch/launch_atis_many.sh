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
dev_frac=0.1
for ds in 0 1 2 3 4
do
  for ms in 0 1
  do
#ds=2
#ms=0
    cl run :src :lib :evaluator :atis :atis-db "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention --stats-file stats.json -l --domain atis -a double:2000 --dev-frac ${dev_frac} --dev-seed ${ds} --model-seed ${ms} --train-data atis/atis_train.tsv --save-file params" --request-docker-image robinjia/robinjia-codalab --request-queue jag --request-cpus $num_cpu -n atis_train -d "double:2000, dev-seed ${ds}, model-seed ${ms}"
    sleep 1
  done
done
