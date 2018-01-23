import cPickle
import gzip
import os
import numpy
import theano


def prepare_data(seqs, labels, maxlen=None):

    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    # find max charlengths
    maxcharlengths = []
    for s in seqs:
        charlengths = [len(w) -3 for w in s]
        maxcharlengths.append(numpy.max(charlengths))

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)

    maxlen = numpy.max(lengths)
    maxcharlen = numpy.max(maxcharlengths)

    n_feats = 3  # for word, pos and chunk

    x = numpy.zeros((maxlen, n_samples, n_feats)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)

    x_char = numpy.zeros((maxlen, maxcharlen, n_samples)).astype('int64')
    x_char_mask = numpy.zeros((maxlen, maxcharlen, n_samples)).astype(theano.config.floatX)

    for idx, s in enumerate(seqs):
        s_word = map(lambda x: x[:n_feats], s)
        x[:lengths[idx], idx, :n_feats] = numpy.array(s_word)
        x_mask[:lengths[idx], idx] = 1.

        for cidx, c in enumerate(s):
            c_ids = c[n_feats:]
            charlen = len(c_ids)

            x_char[cidx, :charlen, idx] = c_ids
            x_char_mask[cidx, :charlen, idx] = 1.

    y = numpy.zeros((maxlen, n_samples)).astype('int64')

    for idx, s in enumerate(labels):
        y[:lengths[idx], idx] = s

    return x, x_mask, x_char, x_char_mask, y


def get_dataset_file(dataset, default_dataset, origin):

    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == default_dataset:
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == default_dataset:
        import urllib
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)
    return dataset


def load_data(path="kornerdatasetdb.pkl", n_words=100000, valid_portion=0.1, maxlen=None, sort_by_len=True):

    path = "./kornerdatasetdb.pkl"
    path = get_dataset_file(path, "kornerdatasetdb.pkl", None)

    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    train_set = cPickle.load(f)
    valid_set = cPickle.load(f)
    test_set = cPickle.load(f)
    f.close()

    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

    # split training set into validation set
    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set
    test_set_x, test_set_y = test_set

    def remove_unk(x):
        for sen in x:
            for w in sen:
                # for words
                w[0] = 1 if w[0] >= n_words else w[0]
        return x

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)
    test_set_x = remove_unk(test_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)

        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)

    return train, valid, test


def load_dict_data(path="kornerdatasetdb.dict.pkl"):
    f = open(path, 'rb')

    word_dict = cPickle.load(f)
    tag_dict = cPickle.load(f)
    feat_dict = cPickle.load(f)
    char_dict = cPickle.load(f)

    return word_dict, tag_dict, feat_dict, char_dict