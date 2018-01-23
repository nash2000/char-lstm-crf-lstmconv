# char-lstm-crf-theano

Theano implementation of ***character-based LSTM CRFs*** presented in the paper [Improving LSTM CRFs Using Character-based Compositions for Korean Named Entity Recognition]().

This neural model combines two types of character-based compositional word representations, one based on convolutional neural networks ([ConvNets](http://proceedings.mlr.press/v32/santos14.html)) and another on [bidirectional LSTMs](http://www.cs.cmu.edu/~lingwang/papers/emnlp2015.pdf), building a ***hybrid word representation***.
The hybrid word representation is obtained by separately apply LSTM-based and ConvNet-based compositions to input character vectors and concatenate the resulting compositional morpheme vectors to finally generate the hybrid representation of a morpheme

Much of the base code is from 



