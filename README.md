# Neural Network Semantic Parsing

Example usage:

    python src/py/main.py -d 100 -i 50 -o 50 -t 20 -p attention -c lstm -m attention -u 1 --train-data data/geo880/processed-lesscopy/geo880_train600.tsv --dev-frac 0.2 --save-file params > out_file

# Notes
The geoquery executor code was taken from 
[Percy Liang's PhD Thesis](http://cs.stanford.edu/~pliang/papers//dcs-thesis2011.pdf),
whose code can be downloaded [here](http://cs.stanford.edu/~pliang/papers/software/dcs.zip).

# Dependencies
## Core
* Python 2.7
* [numpy](http://www.numpy.org/)
* [Theano](http://deeplearning.net/software/theano/)
* [dill](https://pypi.python.org/pypi/dill): a better version of the pickle module

## Additional
* [bottle](http://bottlepy.org/docs/dev/index.html): for browser visualizations
* Java 7: for geoquery and regex evaluation
* [automaton](http://mvnrepository.com/artifact/dk.brics.automaton/automaton/1.11-8): for regex evaluation
* Scala 2.9.0.1: for geoquery denotation computation

Other dependencies are from Percy's thesis--I'll figure out how to deal with them later.
