# Neural Network Semantic Parsing

Example usage:

    python src/py/main.py -d 80 -i 20 -o 20 -t 10 -c lstm -m encoderdecoder --train-data data/overnight/processed/publications_train.tsv --dev-data data/overnight/processed/publications_dev.tsv

# Notes
The geoquery executor code was taken from 
[Percy Liang's PhD Thesis](http://cs.stanford.edu/~pliang/papers//dcs-thesis2011.pdf),
whose code can be downloaded [here](http://cs.stanford.edu/~pliang/papers/software/dcs.zip).

# Dependencies
* Python 2.7
  * [Theano](http://deeplearning.net/software/theano/)
* Java 7
  * [automaton](http://mvnrepository.com/artifact/dk.brics.automaton/automaton/1.11-8)
* Scala 2.9.0.1

Other dependencies are from Percy's thesis--I'll figure out how to deal with them later.
