#!/usr/bin/perl
$embdir = "data/embedding/glove100d";
$dataset = "kornerdatasetlexicondb";
$modeldirname = "models/$dataset";

$pgm = "mkdir -p $modeldirname";
print $pgm, "\n";
system($pgm);

$dim_word_proj = 100;
#$pgm = "THEANO_FLAGS=\'device=gpu1,floatX=float32\' python cclab/bilstmcrf/pretrained_embedding_mapper.py -j $dim_word_proj $dataset.dict.pkl $embdir/embeddings.txt $embdir/words.lst $dataset.word.pretrained.npz";
$pgm = "THEANO_FLAGS=\'device=cpu,floatX=float32\' python cclab/bilstmcrf/pretrained_embedding_mapper.py -j $dim_word_proj $dataset.dict.pkl $embdir/embeddings.txt $embdir/words.lst $dataset.word.pretrained.npz";
print $pgm, "\n";
#system($pgm);

$dim_lstm_hidden_proj = 250;
#$dim_lstm_hidden_proj = 500;
#$dim_lstm_hidden_proj = 750;
#$dim_lstm_hidden_proj = 1000;
$context_size = 3;

$pgm = "THEANO_FLAGS=\'device=gpu2,floatX=float32\' python cclab/bilstmcrf/bilstmcrf_charlstmconv_lexicon.py -j $dim_word_proj -c $context_size -h $dim_lstm_hidden_proj -d $dataset -p $modeldirname -w $dataset.word.pretrained.npz";
print $pgm, "\n";
system($pgm);

#$pgm = "python cclab/theano/tagger/bidirectionallstm_crf_tagger_feat_change_cnn_char_pretrained_full_dropout_directf.py -j $dim_word_proj -c $context_size -h $dim_lstm_hidden_proj  -t $modeldirname/model_output_feat_char.dat -d $dataset -p $modeldirname";
$pgm = "python cclab/bilstmcrf/bilstmcrf_charlstmconv_lexicon.py -j $dim_word_proj -c $context_size -h $dim_lstm_hidden_proj  -t $modeldirname/model_output_feat_char.dat -d $dataset -p $modeldirname";
print $pgm, "\n";
system($pgm);

$pgm = "python cclab/bilstmcrf/model_prediction_result_print.py $modeldirname/model_output_feat_char.dat";
print $pgm, "\n";
system($pgm);


