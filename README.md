# char-lstm-crf-theano

Theano implementation of ***character-based LSTM CRFs*** presented in the paper [Improving LSTM CRFs Using Character-based Compositions for Korean Named Entity Recognition]().

This neural model combines two types of character-based compositional word representations, one based on convolutional neural networks ([ConvNets](http://proceedings.mlr.press/v32/santos14.html)) and another on [bidirectional LSTMs](http://www.cs.cmu.edu/~lingwang/papers/emnlp2015.pdf), building a ***hybrid word representation***.
The hybrid word representation is obtained by separately apply LSTM-based and ConvNet-based compositions to input character vectors and concatenate the resulting compositional morpheme vectors to finally generate the hybrid representation of a morpheme

Much of base codes come from the [LSTM code](http://deeplearning.net/tutorial/lstm.html) in the Theano tutorial. 

### Data
Data should be placed at the `data/kornerdataset` directory, split into `train.txt`, `valid.txt`, and `test.txt'.

Similar to the format of CoNLL-2002 shared task, all data files contain a single morpheme per line with it associated named entity tag in the IOB2 format. Sentences are separated by empty lines. Each line corresponding to a single morpheme consists of the folloiwng four fields:

```
1) Morpheme info -- the surface form of a morpheme associated with its POS tag

2) POS tag -- The POS tag of a morpheme

3) Word space info (BI notation) -- The 'B' tag means that the word spacing unit appears just before the current morpheme. The 'I' tag indicates that the work spacing unit is not given betweeen the previous morpheme and the current morpheme. 

4) NER tag -- The NER tag of a morpheme (folloiwng IOB2 notation)

```


