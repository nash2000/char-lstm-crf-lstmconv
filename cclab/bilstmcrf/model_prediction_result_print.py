'''
Build a tweet sentiment analyzer
'''
from collections import OrderedDict
import cPickle as pkl
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import getopt


def usage():
    print >>sys.stderr, "usage: [model prediction filename]"


def print_pred_results(pred_results, pathto):
    f = open(pathto, "w")

    for local_results in pred_results:
        for wordstr, tagstr in local_results:
            print >>f, "%s\t%s" % (wordstr, tagstr)
        print >>f

    f.close()

if __name__ == '__main__':

    try:
        opts, args = getopt.getopt(sys.argv[1:], "a:v:t:e:l:m:d:n:k:u", ["help", "param="])
    except getopt.GetoptError, err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        #usage()
        usage()
        sys.exit(2)

    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o == "-t":
            modeltype = a
        elif o == "-a":
            actitype = a
        elif o == '-m':
            modeliter = a
        else:
            assert False, "unhandled option"

    if len(args) != 1:
        usage()
        sys.exit(-1)



    #model = numpy.load(args[0])
    f = open(args[0], 'rb')

    train_pred_results = pkl.load(f)
    dev_pred_results = pkl.load(f)
    test_pred_results = pkl.load(f)


    print_pred_results(train_pred_results, args[0]+ ".train.pred")
    print_pred_results(dev_pred_results, args[0] + ".dev.pred")
    print_pred_results(test_pred_results, args[0] + ".test.pred")


    f.close()



