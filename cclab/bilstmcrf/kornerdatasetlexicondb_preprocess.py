import numpy
import cPickle as pkl
from collections import OrderedDict
import glob
import os
import sys
from subprocess import Popen, PIPE

### load pretrained embedding file

def get_pretrained_embedding(embedding_filename, word_filename):
    ###
    f = open(embedding_filename, "r")
    f2 = open(word_filename, "r")

    model = {}

    for line in f:
        line = line.strip()

        line2 = f2.readline().strip()

        fields = line.split()

        data = map(lambda x: float(x), fields)

        model[line2] = numpy.array(data)

        #model[line2] = 1

    return model

def read_corpus(path):
    sentences = []

    f = open(path, "r")

    tokens = []

    for line in f:
        line = line.strip()

        if line == "":
            sentences.append(tokens)
            tokens = []

            continue

        fields = line.split()

        word = fields[0]
        postag = fields[1]
        chunktag = fields[2]
        syllfeats = fields[3:-1]
        tag = fields[-1]

        chars = list(word.decode('utf-8'))
        #chars = list(word)
        ##print word, tag
        #tokens.append((word.lower(),tag))
        tokens.append((word,tag, [postag, chunktag], syllfeats, chars ))

    return sentences


def count2dict(givencount):
    counts = givencount.values()
    keys = givencount.keys()

    sorted_idx = numpy.argsort(counts)[::-1]

    worddict = dict()

    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx+2  # leave 0 and 1 (UNK)

        #print keys[ss], idx+2

    #print numpy.sum(counts), ' total words ', len(keys), ' unique words'
    print >>sys.stderr, numpy.sum(counts), ' total words ', len(keys), ' unique words'

    return worddict


##############################
def build_dict(list_of_sentences, senna_dict):
    #sentences = read_corpus(path)

    print 'Building dictionary..',
    wordcount = dict()
    tagcount = dict()

    charcount = dict()
    syllfeatcount = dict()
    featcounts = None

    train = True

    for sentences in list_of_sentences:


        for words in sentences:
            for w, t, feats, syllfeats, chars in words:
                if train:
                    if w not in wordcount:
                        wordcount[w] = 1
                    else:
                        wordcount[w] += 1

                    if t not in tagcount:
                        tagcount[t] = 1
                    else:
                        tagcount[t] += 1

                    if featcounts is None:
                        featcounts = []
                        for i in range(len(feats)):
                            featcounts.append({})

                    featid = 0

                    for f in feats:
                        if f not in featcounts[featid]:
                            featcounts[featid][f] = 1
                        else:
                            featcounts[featid][f] += 1

                        featid += 1

                    ## for characters
                    for c in chars:
                        if c not in charcount:
                            charcount[c] = 1
                        else:
                            charcount[c] += 1

                    ## for syllables
                    for c in syllfeats:
                        if c not in syllfeatcount:
                            syllfeatcount[c] = 1
                        else:
                            syllfeatcount[c] += 1

                else:
                    if w in senna_dict:
                        if w not in wordcount:
                            wordcount[w] = 1
                        else:
                            wordcount[w] += 1

                    if t not in tagcount:
                        tagcount[t] = 1
                    else:
                        tagcount[t] += 1

                    for c in chars:
                        if c not in charcount:
                            charcount[c] = 1
                        else:
                            charcount[c] += 1

        train = False


    worddict = count2dict(wordcount)

    tagdict = count2dict(tagcount)

    featdicts = map(lambda count: count2dict(count), featcounts)


    syllfeatdict = count2dict(syllfeatcount)

    chardict = count2dict(charcount)


    return worddict, tagdict, featdicts, syllfeatdict,  chardict


def grab_data(sentences, worddict, tagdict, featdict, syllfeatdict, chardict):

    seqs = [None] * len(sentences)
    seqs_y =  [None] * len(sentences)
    seqs_f =  [None] * len(sentences)

    seqs_sf =  [None] * len(sentences)
    seqs_c =  [None] * len(sentences)

    for idx, sentence in enumerate(sentences):
        words = map(lambda sent: sent[0], sentence)
        tags = map(lambda sent: sent[1], sentence)
        feats = map(lambda sent: sent[2], sentence)
        syllfeats = map(lambda sent: sent[3], sentence)
        chars = map(lambda sent: sent[4], sentence)

        #print idx, words
        #print words, tags
        seqs[idx] = [worddict[w] if w in worddict else 1 for w in words]

        ##print tags
        seqs_y[idx] = [tagdict[w] if w in tagdict else tagdict["NN"] for w in tags]

        seqs_f[idx] = []

        for f in feats:
            fids = []
            for featidx in range(len(featdict)):
                fid = featdict[featidx][ f[featidx] ] if f[featidx] in featdict[featidx] else 1
                fids.append(fid)
            #local_seqs_f = [ featdict[featidx][ f[featidx] ] if f[featidx] in featdict[featidx] else 1 for f in feats]
            seqs_f[idx].append(fids)


        seqs_sf[idx] = []

        for localsyllfeats in syllfeats:

            cids = [syllfeatdict[c] if c in syllfeatdict else 1 for c in localsyllfeats]
            #local_seqs_f = [ featdict[featidx][ f[featidx] ] if f[featidx] in featdict[featidx] else 1 for f in feats]
            seqs_sf[idx].append(cids)


        seqs_c[idx] = []

        for localchars in chars:

            cids = [chardict[c] if c in chardict else 1 for c in localchars]
            #local_seqs_f = [ featdict[featidx][ f[featidx] ] if f[featidx] in featdict[featidx] else 1 for f in feats]
            seqs_c[idx].append(cids)

        #seqs_f[idx][featidx] = local_seqs_f

        #print seqs[idx]

    new_seqs_x = []
    new_seqs_y = []
    for x, feat, syllfeats, chars, y in zip(seqs, seqs_f, seqs_sf, seqs_c, seqs_y):
        merged_s_f_c = map(lambda m: [m[0]]+m[1]+m[2]+m[3], zip(x,feat, syllfeats, chars))

        #print merged_s_f
        #print merged_s_f

        new_seqs_x.append(merged_s_f_c)
        new_seqs_y.append(y)

    return new_seqs_x, new_seqs_y

def main():
    dirpath = "data/kornerdatasetlexicon"

    #dirpath = "."

    train_path  =   os.path.join(dirpath, "train.txt")
    dev_path    =   os.path.join(dirpath, "dev.txt")
    test_path   =   os.path.join(dirpath, "test.txt")

    #sennadirpath = "data/embedding/glove50d"
    sennadirpath = "data/embedding/glove100d"

    embedding_path = os.path.join(sennadirpath, "embeddings.txt")
    word_path = os.path.join(sennadirpath, "words.lst")

    senna_dict = get_pretrained_embedding(embedding_path, word_path)

    print >>sys.stderr, train_path
    train_sentences = read_corpus(train_path)

    print >>sys.stderr, dev_path
    dev_sentences   = read_corpus(dev_path)

    print >>sys.stderr, test_path
    test_sentences  = read_corpus(test_path)

    word_dictionary, tag_dictionary, feat_dictionary, syllfeat_dictionary, char_dictionary = build_dict([train_sentences, dev_sentences, test_sentences], senna_dict)

    train_x, train_y = grab_data(train_sentences, word_dictionary, tag_dictionary, feat_dictionary, syllfeat_dictionary, char_dictionary)
    dev_x, dev_y = grab_data(dev_sentences, word_dictionary, tag_dictionary, feat_dictionary, syllfeat_dictionary, char_dictionary)
    test_x, test_y = grab_data(test_sentences, word_dictionary, tag_dictionary, feat_dictionary, syllfeat_dictionary, char_dictionary)

    print >>sys.stderr, "num training sentences:", len(train_x)
    print >>sys.stderr, "num dev sentences:", len(dev_x)
    print >>sys.stderr, "num test sentences:", len(test_x)

    f = open('kornerdatasetlexicondb.pkl', 'wb')
    pkl.dump((train_x, train_y), f, -1)
    pkl.dump((dev_x, dev_y), f, -1)
    pkl.dump((test_x, test_y), f, -1)
    f.close()

    f = open('kornerdatasetlexicondb.dict.pkl', 'wb')
    pkl.dump(word_dictionary, f, -1)
    pkl.dump(tag_dictionary, f, -1)
    pkl.dump(feat_dictionary, f, -1)
    pkl.dump(syllfeat_dictionary, f, -1)
    pkl.dump(char_dictionary, f, -1)

    f.close()

if __name__ == '__main__':
    main()