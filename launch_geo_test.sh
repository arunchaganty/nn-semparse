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
cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32 python src/py/main.py -d $d -i $i -o $i -p none -u 1 -t 25,5,5 -c lstm -m encoderdecoder -k 10 --stats-file stats.json --domain geoquery --train-data geo880-lesscopy/geo880_train600.tsv --dev-data geo880-lesscopy/geo880_test280.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_test -d '600, no augmentation' $flags
sleep 1
cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32 python src/py/main.py -d $d -i $i -o $i -p none -u 1 -t 25,5,5 -c lstm -m encoderdecoder -k 10 --stats-file stats.json --domain geoquery --train-data geo880-augmented/geo880_train600_augpcfg300.tsv --dev-data geo880-lesscopy/geo880_test280.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_test -d '600, augment 300' $flags
sleep 1
cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32 python src/py/main.py -d $d -i $i -o $i -p none -u 1 -t 25,5,5 -c lstm -m attention -k 10 --stats-file stats.json --domain geoquery --train-data geo880-lesscopy/geo880_train600.tsv --dev-data geo880-lesscopy/geo880_test280.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_test -d '600, no augmentation' $flags
sleep 1
cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32 python src/py/main.py -d $d -i $i -o $i -p none -u 1 -t 25,5,5 -c lstm -m attention -k 10 --stats-file stats.json --domain geoquery --train-data geo880-augmented/geo880_train600_augpcfg300.tsv --dev-data geo880-lesscopy/geo880_test280.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_test -d '600, augment 300' $flags
sleep 1
#cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention -k 10 --stats-file stats.json --domain geoquery --train-data geo880-lesscopy/geo880_train100.tsv --dev-data geo880-lesscopy/geo880_test280.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_test -d '100, no augmentation' $flags
#sleep 1
#cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention -k 10 --stats-file stats.json --domain geoquery --train-data geo880-augmented/geo880_train100_augpcfg50.tsv --dev-data geo880-lesscopy/geo880_test280.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_test -d '100, augment 50' $flags
#sleep 1
#cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention -k 10 --stats-file stats.json --domain geoquery --train-data geo880-lesscopy/geo880_train200.tsv --dev-data geo880-lesscopy/geo880_test280.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_test -d '200, no augmentation' $flags
#sleep 1
#cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention -k 10 --stats-file stats.json --domain geoquery --train-data geo880-augmented/geo880_train200_augpcfg100.tsv --dev-data geo880-lesscopy/geo880_test280.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_test -d '200, augment 100' $flags
#sleep 1
#cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention -k 10 --stats-file stats.json --domain geoquery --train-data geo880-lesscopy/geo880_train300.tsv --dev-data geo880-lesscopy/geo880_test280.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_test -d '300, no augmentation' $flags
#sleep 1
#cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention -k 10 --stats-file stats.json --domain geoquery --train-data geo880-augmented/geo880_train300_augpcfg150.tsv --dev-data geo880-lesscopy/geo880_test280.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_test -d '300, augment 150' $flags
#sleep 1
#cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention -k 10 --stats-file stats.json --domain geoquery --train-data geo880-lesscopy/geo880_train400.tsv --dev-data geo880-lesscopy/geo880_test280.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_test -d '400, no augmentation' $flags
#sleep 1
#cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention -k 10 --stats-file stats.json --domain geoquery --train-data geo880-augmented/geo880_train400_augpcfg200.tsv --dev-data geo880-lesscopy/geo880_test280.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_test -d '400, augment 200' $flags
#sleep 1
#cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention -k 10 --stats-file stats.json --domain geoquery --train-data geo880-lesscopy/geo880_train500.tsv --dev-data geo880-lesscopy/geo880_test280.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_test -d '500, no augmentation' $flags
#sleep 1
#cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention -k 10 --stats-file stats.json --domain geoquery --train-data geo880-augmented/geo880_train500_augpcfg250.tsv --dev-data geo880-lesscopy/geo880_test280.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_test -d '500, augment 250' $flags
#sleep 1
cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention -k 10 --stats-file stats.json --domain geoquery --train-data geo880-lesscopy/geo880_train600.tsv --dev-data geo880-lesscopy/geo880_test280.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_test -d '600, no augmentation' $flags
sleep 1
cl run :src :lib :evaluator :geo880-lesscopy :geo880-augmented "OMP_NUM_THREADS=4 THEANO_FLAGS=blas.ldflags=-lopenblas,device=${device},floatX=float32 python src/py/main.py -d $d -i $i -o $i -p attention -u 1 -t 25,5,5 -c lstm -m attention -k 10 --stats-file stats.json --domain geoquery --train-data geo880-augmented/geo880_train600_augpcfg300.tsv --dev-data geo880-lesscopy/geo880_test280.tsv --save-file params" --request-queue jag --request-cpus $num_cpu -n geoquery_test -d '600, augment 300' $flags
