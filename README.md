# hep-calo-generative-modeling

NYU data science project to simulate calorimeter readings using neural networks.

Charlie Guthrie (cdg356@nyu.edu), Israel Malkin (im965@nyu.edu), and Alex Pine (akp258@nyu.edu), under supervision from Prof. Kyle Cranmer (kyle.cranmer@nyu.edu).

## Source Files

Name                        | Description                        | Location
----------------------------|----------------------------------- | ----------------------
Flat RNN Model | RNN model. Trains a model and generates samples for evaluation. Requires Keras 2.0. | [scripts/rnn_flat/rnn_flat.py](https://github.com/pinesol/hep-calo-generative-modeling/tree/master/scripts/rnn_flat/rnn_flat.py)
MLP GAN Model | Wasserstein GAN model, simple MLP model. Trains a model and generates samples for evaluation. Requires Keras 1.5. | [scripts/spiral_gan/small.py](https://github.com/pinesol/hep-calo-generative-modeling/tree/master/scripts/spiral_gan/small.py)
LAGAN Model | Wasserstein GAN model, Location-Aware GAN. Trains a model and generates samples for evaluation. Requires Keras 1.5. | [scripts/spiral_gan/local.py](https://github.com/pinesol/hep-calo-generative-modeling/tree/master/scripts/spiral_gan/local.py)
Data Loader | Code to load and clean the calorimeter data. | [scripts/data_loader.py](https://github.com/pinesol/hep-calo-generative-modeling/tree/master/scripts/data_loader.py)
Eval | Code to generate evaluation charts given a file of generated images. | [scripts/eval.py](https://github.com/pinesol/hep-calo-generative-modeling/tree/master/scripts/eval.py)
