# char-lstm-crf-theano

Theano implementation of ***character-based LSTM CRFs*** presented in the paper [Improving LSTM CRFs Using Character-based Compositions for Korean Named Entity Recognition]().

The neural model is based on a hybrid representation that combines both LSTM-based and ConvNet-based compositional word vectors. 
The hybrid word representation is obtained by separately apply LSTM-based and ConvNet-based compositions to input character vectors and concatenate the resulting compositional morpheme vectors to finally generate the hybrid representation of a morpheme

Much of the base code is from 



