import numpy
import os
import sys
import getopt
import cPickle
from theano import config

def usage():
    print "usage ... (option) [corpusdbfilename] [sennaembeddingfilename] [wordfilename] [outputfilename]"


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

    return model

def build_pretrained_matrix(num_words, word_dict, embedding_model, dim_word_proj=50):
    #Wemd = numpy.zeros((num_words, 50), dtype=float)

    randn = numpy.random.rand(num_words, dim_word_proj)
    Wemd = (0.01 * randn).astype(config.floatX)

    num_matched = 0

    for str, id in word_dict.items():

        if id < num_words and embedding_model.has_key(str) is True:
            emb = embedding_model[str]
            Wemd[id] = emb
            num_matched += 1

    print >>sys.stderr, "num_total_Words:", len(word_dict)
    print >>sys.stderr, "num_Matched: ", num_matched

    print >>sys.stderr, "coverage: ", float(num_matched)/float(len(word_dict))

    return Wemd



if __name__ == '__main__':

    num_words = 100000
    dim_word_proj = 50

    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:j:v", ["help", "param="])
    except getopt.GetoptError, err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        #usage()
        usage()
        sys.exit(2)

    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o == "-j":
            dim_word_proj = int(a)
        elif o == '-n':
            num_words = int(a)
        else:
            assert False, "unhandled option"

    if len(args) != 4:
        usage()
        sys.exit(-1)

    f = open(args[0], 'rb')
    word_dict = cPickle.load(f)
    f.close()
    embedding_model = get_pretrained_embedding(args[1], args[2])
    Wemb = build_pretrained_matrix(num_words, word_dict, embedding_model, dim_word_proj)

    print >>sys.stderr, "Wemb is stored at %s " % (args[3])

    numpy.savez(args[3], Wemb=Wemb)
