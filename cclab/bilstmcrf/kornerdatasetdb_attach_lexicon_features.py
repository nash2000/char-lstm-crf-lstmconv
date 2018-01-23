import cPickle as pkl
import os
import sys
import getopt
import glob
from collections import defaultdict


def read_corpus(path):
    f = open(path, "r")
    sents = []
    sent = []
    for line in f:
        line = line.strip()
        if line == "":
            sents.append(sent)
            sent = []
            continue
        fields = line.split()
        word = fields[0].split("/")[0]
        fields[0] = word
        sent.append(fields)
    return sents


def write_corpus(path, sents):
    f = open(path, "w")

    for sent in sents:
        for tokens in sent:
            tokens[0] = tokens[0] + "/" + tokens[1]
            f.write(" ".join(tokens) + "\n")
        f.write("\n")
    f.close()


def load_lexicon_dict(lex_dir):
    ners = ['PS', 'LC', 'OG']
    lexicon_dict = defaultdict(dict)

    for ner in ners:
        lex_filename = os.path.join(lex_dir, "%s.lexicon" % ner)
        lines = open(lex_filename).readlines()

        for line in lines:
            element = line.strip()
            element = tuple(element.split())
            lexicon_dict[ner][element] = 1

    return lexicon_dict


def lexicon_feature_maximal_matching(ner_dict, mophms, tags, plo):
    length = len(mophms)
    lexicons = ["O" for i in range(length)]

    for jump in range(1, length + 1):
        for idx in range(0, length):
            if idx + jump > length:
                continue
            if jump == 1 and tags[idx][0] == "V":
                continue
            if jump == 1 and len(mophms[idx].decode("utf-8")) == 1:
                continue

            key = tuple(mophms[idx:idx+jump])
            if ner_dict.has_key(key):
                if jump == 1:
                    lexicons[idx] = "S-" + plo.upper()
                else:
                    lexicons[idx] = "B-" + plo.upper()
                    lexicons[idx+jump-1] = "E-" + plo.upper()
                    for i in range(idx+1, idx+jump-1):
                        lexicons[i] = "I-" + plo.upper()

    return lexicons


def append_lexicon_features(sents_list, ner_dicts):
    ners = ['PS', 'LC', 'OG']
    new_sents_list = []
    for sentence in sents_list:
        total_lexicons = [[] for i in range(len(sentence))]
        mophms = map(lambda sent: sent[0], sentence)
        tags = map(lambda sent: sent[1], sentence)

        for ner in ners:
            ner_dict = ner_dicts[ner]
            lexicons = lexicon_feature_maximal_matching(ner_dict, mophms, tags, ner)
            for i, (total_lexicon, lexicon) in enumerate(zip(total_lexicons,lexicons)):
                total_lexicons[i].append(lexicon)

        new_sent = []
        for tokens, lexicons in zip(sentence, total_lexicons):
            tokens = list(tokens)
            fields = tokens[:3] + lexicons + [tokens[-1]]
            new_sent.append(fields)
        new_sents_list.append(new_sent)

    return new_sents_list


def main():
    dirpath = None
    outdirpath = None
    lexicondir = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:o:l:", ["help", "param="])
    except getopt.GetoptError, err:
        print str(err)
        sys.exit(2)

    for o, a in opts:
        if o == "-d":
            dirpath = a
        elif o == "-o":
            outdirpath = a
        elif o == "-l":
            lexicondir = a
        else:
            assert False, "unhandled option"

    if not os.path.exists(outdirpath):
        os.makedirs(outdirpath)

    lexicon_dict = load_lexicon_dict(lexicondir)

    datasets = ['train.txt', 'dev.txt', 'test.txt']
    for filename in datasets:
        datapath = os.path.join(dirpath, filename)
        outdatapath = os.path.join(outdirpath, filename)
        sents = read_corpus(datapath)
        lexicon_attached_sents = append_lexicon_features(sents, lexicon_dict)
        write_corpus(outdatapath, lexicon_attached_sents)


if __name__ == '__main__':
    main()
