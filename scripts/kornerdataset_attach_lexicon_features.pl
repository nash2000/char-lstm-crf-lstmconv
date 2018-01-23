#!/usr/bin/perl
$dirpath = "data/kornerdataset";
#$outdirpath = "data/kornerdatasetlexicon2";
$outdirpath = "data/kornerdatasetlexicon";
$lexicondir = "data/lexicons";

$pgm = "mkdir -p $outdirpath";
print $pgm, "\n";
system($pgm);

$pgm = "python cclab/bilstmcrf/kornerdatasetdb_attach_lexicon_features.py -d $dirpath -o $outdirpath -l $lexicondir";
print $pgm, "\n";
system($pgm);
