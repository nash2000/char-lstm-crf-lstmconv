# char-lstm-crf-theano

Theano implementation of ***character-based LSTM CRFs*** presented in the paper [Improving LSTM CRFs Using Character-based Compositions for Korean Named Entity Recognition]().

This neural model combines two types of character-based compositional word representations, one based on convolutional neural networks ([ConvNets](http://proceedings.mlr.press/v32/santos14.html)) and another on [bidirectional LSTMs](http://www.cs.cmu.edu/~lingwang/papers/emnlp2015.pdf), building a ***hybrid word representation***.
The hybrid word representation is obtained by separately apply LSTM-based and ConvNet-based compositions to input character vectors and concatenate the resulting compositional morpheme vectors to finally generate the hybrid representation of a morpheme

Much of base codes come from the [LSTM code](http://deeplearning.net/tutorial/lstm.html) in the Theano tutorial. 

# Data
Data should be placed at the `data/kornerdataset` directory, split into `train.txt`, `valid.txt`, and `test.txt'.

Similar to the format of CoNLL-2002 shared task, all data files contain a single morpheme per line with it associated named entity tag in the IOB2 format. Sentences are separated by empty lines. Each line corresponding to a single morpheme consists of the folloiwng four fields:

1. **Morpheme info**: The surface form of a morpheme associated with its POS tag
2. **POS tag**: The POS tag of a morpheme
3. **Word space info** (with the BI notation): The 'B' tag means that the word spacing unit appears just before the current morpheme. The 'I' tag indicates that the work spacing unit is not given betweeen the previous morpheme and the current morpheme. 
4. **NER tag**: The NER tag of a morpheme (following IOB2 notation)


An example of data is given as follows: 

```
프랭크/NNP      NNP     B       B-PS
로빈슨/NNP      NNP     B       I-PS
감독/NNG        NNG     B       O
,/SP    SP      I       O
"/SS    SS      B       O
김선우/NNP      NNP     I       B-PS
정말/MAG        MAG     B       O
잘/MAG  MAG     B       O
던졌/VV~EP      VV~EP   B       O
다/EF   EF      I       O
"/SS    SS      I       O
```

#### Pretrained embedding vectors

Data files for pretrained embedding vectors should be located at the directory like `data/embedding/XXX`, which include `embeddings.txt` and `words.lst` where `XXX` is the subdirectory name for the embedding method and the dimensionality.

For Korean NER task, data files for pretrained embeddings vectors of Korean are located at `data/embedding/glove100d`


# Preprocessing Data by Adding Lexicon Features

To add lexicon features, lexicon files should be located in the `data/lexicons`, split into `LC.lexicon`, `OG.lexicon`, `PS.lexicon`
where our Korean NER task have four NER tags: 

- PS: Person, LC: Location, OG: Organization, DT: Date/Time


Run the following commands: 
``` sh
mkdir -p data/kornerdatasetlexicon
python cclab/bilstmcrf/kornerdatasetdb_attach_lexicon_features.py -d data/kornerdataset -o data/kornerdatasetlexicon -l data/lexicons
```
or equivalently you can run the following script
``` sh
scripts/kornerdataset_attach_lexicon_features.pl
```

Then, the command will create the preproceed data in `data/kornerdatasetlexicon` where each line of original data is augmented with lexicon features, consistsing of seven fields as follows: 

 - 1st-3rd fields: **Morpheme info**, **POS tag**, **Word space info**
 - 4th-6th fields: **Lexicon features**
 - 7th field: **NER tag**
 
An example of processed data is given as follows: 

```
프랭크/NNP NNP B B-PS S-LC O B-PS
로빈슨/NNP NNP B E-PS S-LC S-OG I-PS
감독/NNG NNG B O O O O
,/SP SP I O O O O
"/SS SS B O O O O
김선우/NNP NNP I S-PS O O B-PS
정말/MAG MAG B O O O O
잘/MAG MAG B O O O O
던졌/VV~EP VV~EP B O O O O
다/EF EF I O O O O
"/SS SS I O O O O
```


# Training a model
Check first whether all data files exist in `data/kornerdatasetlexicon`

Training consists of three processing steps, as follows:

- Instead of reading text files directly, we first convert data (train/dev/test) to numpy arrays and store them in a pickle file and create a dictionary pickle that stores all the morphemes in data. 

#### 1. Build pickle files for data

Run the following command to build pickle files.

``` sh
python cclab/bilstmcrf/kornerdatasetlexicondb_preprocess.py
```

This command will create `kornerdatasetlexicondb.pkl` (the pickle file for data) and `kornerdatasetlexicondb.dict.pkl` (the pickle file for dictionary).


#### 2. Filter and align pretrained embedding vectors 

Run the following command to filter and align pretrained embedding vectors 

- Usually, the size of pretrained embedding vectors are significantly larger than those provided in data. To make the training efficient, we need to select only morphemes that occur at least once in dataset, among all the pretrained morphemes. In selected files, the morpheme ids should be aligned with those of the morpheme dictionary.

``` sh
mkdir -p models/kornerdatasetlexicondb
python cclab/bilstmcrf/pretrained_embedding_mapper.py -j 100 kornerdatasetlexicondb.dict.pkl data/embedding/glove100d/embeddings.txt data/embedding/glove100d/words.lst kornerdatasetlexicondb.word.pretrained.npz
```
 

The command will create the selected and aligned embedding file `kornerdatasetlexicondb.word.pretrained.npz`

#### 3. Training

We are now ready for training.

Run the command for training a NER model.

``` sh
python cclab/bilstmcrf/bilstmcrf_charlstmconv_lexicon.py -j 100 -c 3 -h 250 -d kornerdatasetlexicondb -p models/kornerdatasetlexicondb -w kornerdatasetlexicondb.word.pretrained.npz
```


During training, you will see the screen like: 
```
Optimization
4230 train examples
250 valid examples
500 test examples
1100 valid freq
1100 save freq
Epoch  0 Update  10 Cost  45.3594551086
Epoch  0 Update  20 Cost  48.3083953857
Epoch  0 Update  30 Cost  41.0771827698
Epoch  0 Update  40 Cost  36.9920349121
Epoch  0 Update  50 Cost  29.4389381409
Epoch  0 Update  60 Cost  33.2070007324
Epoch  0 Update  70 Cost  26.0016994476
Epoch  0 Update  80 Cost  37.4194755554
Epoch  0 Update  90 Cost  28.0124607086
Epoch  0 Update  100 Cost  34.4811019897
Epoch  0 Update  110 Cost  26.8870029449
Epoch  0 Update  120 Cost  29.7778434753
Epoch  0 Update  130 Cost  23.4083709717
Epoch  0 Update  140 Cost  23.3320198059
Epoch  0 Update  150 Cost  25.4277305603
Epoch  0 Update  160 Cost  24.8934879303
Epoch  0 Update  170 Cost  21.8137321472
Epoch  0 Update  180 Cost  24.1368789673
...
```

## Training a model (using a perl script)

Run the perl script that include the three processing steps above.

``` sh
./scripts/bilstmcrf_charlstmconv_lexicon_train.pl
``` 

## Training a model (without lexicon features)
Check first whether all data files exist in `data/kornerdataset`

Run the perl script that include the three processing steps above.

``` sh
./scripts/bilstmcrf_charlstmconv_train.pl
```

Or, run the three processing steps sequentially:

#### 1. Build pickle files for data

``` sh
python cclab/bilstmcrf/kornerdatasetdb_preprocess.py
```

The command will create `kornerdatasetdb.pkl` (the pickle file for data) and `kornerdatasetdb.dict.pkl` (the pickle file for dictionary).


#### 2. Filter and align pretrained embedding vectors 

Run the following command to filter and align pretrained embedding vectors 

``` sh
python cclab/bilstmcrf/pretrained_embedding_mapper.py -j 100 kornerdatasetdb.dict.pkl data/embedding/glove100d/embeddings.txt data/embedding/glove100d/words.lst kornerdatasetdb.word.pretrained.npz
```

The command will create the selected and aligned embedding file `kornerdatasetlexicondb.word.pretrained.npz`

#### 3. Training

We are now ready for training.

Run the command for training a NER model.

``` sh
 python cclab/bilstmcrf/bilstmcrf_charlstmconv.py -j 100 -c 3 -h 250 -d kornerdatasetdb -p models/kornerdatasetdb -w kornerdatasetdb.word.pretrained.npz
```


# Testing a model

##### 1. Apply the trained model to test set

``` sh
python cclab/bilstmcrf/bilstmcrf_charlstmconv_lexicon.py -j 100 -c 3 -h 250  -t models/kornerdatasetlexicondb/model_output_feat_char.dat -d kornerdatasetlexicondb -p models/kornerdatasetlexicondb
```
The tagged results will be stored in the files with prefix `models/kornerdatasetlexicondb/model_output_feat_char.dat'


##### 2. Print the tagged results from the files

``` sh
python cclab/bilstmcrf/model_prediction_result_print.py models/kornerdatasetlexicondb/model_output_feat_char.dat
```

# Evaluation result

Comparison of the best performance results of char-LSTM, char-ConvNet, and
char-LSTM+ConvNet.

- char-LSTM+ConvNet: The proposed hybrid word representation

||chardim=300|chardim=100|
|---|---|---|
|char-LSTM|88.66%|88.30%|
|char-ConvNet|88.27%|88.60%|
|char-LSTM+ConvNet|**89.04**|**89.01%**|
