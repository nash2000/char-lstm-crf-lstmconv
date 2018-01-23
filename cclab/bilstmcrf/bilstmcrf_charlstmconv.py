'''
Build Korean NER tagger using bidirectional LSTM CRFs (lexicon version)
 (reference: lstm.py in Theano )
'''
from collections import OrderedDict
import cPickle as pkl
import sys
import time

import getopt
import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import os
import kornerdatasetdb
import cPickle as pkl

config.exception_verbosity = 'high'


datasets = {'kornerdatasetdb': (kornerdatasetdb.load_data, kornerdatasetdb.prepare_data, kornerdatasetdb.load_dict_data)}


# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)

def numpy_intX(data):
    return numpy.asarray(data, dtype=int)

def build_id2str_dict(dict):

    new_dict ={}

    for key, value in dict.items():
        new_dict[value] = key

    max_value = numpy.max( dict.values() )

    for i in range(max_value):
        if new_dict.has_key(i) == False:
            new_dict[i] = "O"


    return new_dict

def build_tag_segment_type_dict(dict):

    new_dict ={}

    max_value = 0

    for key, value in dict.items():
        if key.startswith("I"):
            new_dict[value] = 1
        elif key.startswith("O"):
            new_dict[value] = 2
        else:
            new_dict[value] = 0

        if max_value < value:
            max_value = value

    for i in range(max_value):
        if new_dict.has_key(i) == False:
            new_dict[i] = 2

    return new_dict


def has_segmentation_problem(tag_dict, dataset=None):
    ####
    if dataset:
        if dataset == 'sejongdb' or dataset == 'conll03db':
            return True

    has_segmentation = False

    for key in tag_dict.keys():
        if key.startswith("B") == False and key.startswith("I") == False and key.startswith("O") == False and key.startswith("S") == False and key.startswith("E") == False:
            return False

    return True


def theano_logsumexp(x, axis=None):
    """
    Compute log(sum(exp(x), axis=axis) in a numerically stable
    fashion.

    Parameters
    ----------
    x : tensor_like
        A Theano tensor (any dimension will do).

    axis : int or symbolic integer scalar, or None
        Axis over which to perform the summation. `None`, the
        default, performs over all axes.

    Returns
    -------
    result : ndarray or scalar
        The result of the log(sum(exp(...))) operation.
    """
    xmax = x.max(axis=axis, keepdims=True)
    xmax_ = x.max(axis=axis)
    return xmax_ + tensor.log(tensor.exp(x - xmax).sum(axis=axis))


def tensor_flip(x):
    x = tensor.tensor3('x')
    f = theano.function([x],x[::-1])

    return f

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def get_dataset(name):
    return datasets[name][0], datasets[name][1], datasets[name][2]


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)



def ortho_weight_extended(ndim, k):

    Ws = []
    for i in range(k):

        W = numpy.random.randn(ndim, ndim)
        u, s, v = numpy.linalg.svd(W)
        #return u.astype(config.floatX)

        Ws.append(u.astype(config.floatX))

    return numpy.concatenate(Ws, axis=0)



def norm_vector(nin, scale=0.01):
    V = scale * numpy.random.randn(nin)
    return V.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def norm_weight_extended(nin, nout, k, scale=0.01, ortho=True):
    Ws = []
    for i in range(k):
        W  = None
        if nout == nin and ortho:
            W = ortho_weight(nin)
        else:
            W = scale * numpy.random.randn(nin, nout)

        Ws.append(W.astype(config.floatX))

    return numpy.concatenate(Ws, axis=0)



def init_params(options, embeddings=None):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding

    if embeddings is not None:
        print >>sys.stderr, "pretrained embeddings are loaded from ", embeddings
        pp = numpy.load(embeddings)

        params['Wemb'] = pp["Wemb"]

    else:
        randn = numpy.random.rand(options['n_words'],
                                  options['dim_word_proj'])
        params['Wemb'] = (0.01 * randn).astype(config.floatX)


    randn = numpy.random.rand(options['n_postags'],
                              options['dim_postag_proj'])
    params['Pemb'] = (0.01 * randn).astype(config.floatX)


    randn = numpy.random.rand(options['n_chunks'],
                              options['dim_chunk_proj'])
    params['Cemb'] = (0.01 * randn).astype(config.floatX)


    randn = numpy.random.rand(options['n_chars'],
                              options['dim_char_proj'])
    params['Semb'] = (0.01 * randn).astype(config.floatX)

    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])

    # for backward
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix="%s_backward"  % (options['encoder']) )
    # for chunk lstm

    params = get_layer(options['encoder_general'])[0](options,
                                              params,
                                              prefix="%s_char" % (options['encoder_general']),
                                              dim_proj_option_str='dim_char_proj')
    # for backward
    params = get_layer(options['encoder_general'])[0](options,
                                              params,
                                              prefix="%s_char_backward"  % (options['encoder_general']),
                                              dim_proj_option_str='dim_char_proj')


    #params['conv_W'] = 0.01 * numpy.random.randn(ortho_weight_extended(options['dim_char_proj'], options['n_context'],
    #                                        options['dim_conv_char_proj']).astype(config.floatX)
    params['conv_W'] = ortho_weight_extended(options['dim_char_proj'], options['n_char_context'])

    # classifier

    # for bidirectional lstm
    params['U'] = 0.01 * numpy.random.randn(options['dim_lstm_hidden_proj'] * 2,
                                            options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    # transition matrix for viterbi

    #params['A'] = 0.01 * numpy.random.randn(options['ydim'],
    #                                        options['ydim']).astype(config.floatX)

    params['A'] = numpy.zeros((options['ydim'],options['ydim'])).astype(config.floatX)

    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns



def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    W = numpy.concatenate([norm_weight(options['dim_proj'], options['dim_lstm_hidden_proj']),
                           norm_weight(options['dim_proj'], options['dim_lstm_hidden_proj']),
                           norm_weight(options['dim_proj'], options['dim_lstm_hidden_proj']),
                           norm_weight(options['dim_proj'], options['dim_lstm_hidden_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([norm_weight(options['dim_lstm_hidden_proj'], options['dim_lstm_hidden_proj']),
                           norm_weight(options['dim_lstm_hidden_proj'], options['dim_lstm_hidden_proj']),
                           norm_weight(options['dim_lstm_hidden_proj'], options['dim_lstm_hidden_proj']),
                           norm_weight(options['dim_lstm_hidden_proj'], options['dim_lstm_hidden_proj'])], axis=1)
    params[_p(prefix, 'U')] = U

    b = numpy.zeros((4 * options['dim_lstm_hidden_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_lstm_hidden_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_lstm_hidden_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_lstm_hidden_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_lstm_hidden_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    #dim_proj = options['dim_proj']
    dim_hidden_proj = options['dim_lstm_hidden_proj']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_hidden_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_hidden_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]



######  lstm for pos tags and chunk

def param_init_lstm_general(options, params, prefix='lstm_char', dim_proj_option_str='dim_char_proj'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    W = numpy.concatenate([ortho_weight(options[dim_proj_option_str]),
                           ortho_weight(options[dim_proj_option_str]),
                           ortho_weight(options[dim_proj_option_str]),
                           ortho_weight(options[dim_proj_option_str])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options[dim_proj_option_str]),
                           ortho_weight(options[dim_proj_option_str]),
                           ortho_weight(options[dim_proj_option_str]),
                           ortho_weight(options[dim_proj_option_str])], axis=1)
    params[_p(prefix, 'U')] = U

    b = numpy.zeros((4 * options[dim_proj_option_str],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params


def lstm_layer_general(tparams, state_below, options, prefix='lstm_char', mask=None, dim_proj_option_str='dim_char_proj'):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options[dim_proj_option_str]))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options[dim_proj_option_str]))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options[dim_proj_option_str]))
        c = tensor.tanh(_slice(preact, 3, options[dim_proj_option_str]))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options[dim_proj_option_str]
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]



# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
#layers = {'lstm': (param_init_lstm, lstm_layer)}

layers = {'lstm': (param_init_lstm, lstm_layer),
          'lstm_general' : ( param_init_lstm_general, lstm_layer_general)}


def sgd(lr, tparams, grads, x, mask, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, x_pos, x_chunk, mask, x_char, mask_char, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, x_pos, x_chunk, mask, x_char, mask_char, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, x_pos, x_chunk, mask, x_char, mask_char, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, x_pos, x_chunk, mask, x_char, mask_char, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def build_model(tparams, options):
    trng = RandomStreams(SEED)



    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.matrix('x', dtype='int64')
    x_char = tensor.tensor3('x_char', dtype='int64')

    x_pos = tensor.matrix('x_pos', dtype='int64')
    x_chunk = tensor.matrix('x_chunk', dtype='int64')

    mask = tensor.matrix('mask', dtype=config.floatX)
    mask_char = tensor.tensor3('mask_char', dtype=config.floatX)

    y = tensor.matrix('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    n_chartimesteps = x_char.shape[1]

    emb_word = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_word_proj']])

    emb_pos = tparams['Pemb'][x_pos.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_postag_proj']])

    emb_chunk = tparams['Cemb'][x_chunk.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_chunk_proj']])


    emb_char =  tparams['Semb'][x_char.flatten()].reshape([n_timesteps,
                                                n_chartimesteps,
                                                n_samples,
                                                options['dim_char_proj']])


    #########################################################
    def _char_step(m_, emb_c_):

        reversed_emb_c_ = emb_c_[::-1]
        reversed_m_ = m_[::-1]

        char_proj = get_layer(options['encoder_general'])[1](tparams, emb_c_, options,
                                            prefix="%s_char" % (options['encoder_general']),
                                            mask=m_,
                                            dim_proj_option_str='dim_char_proj')

        reversed_char_proj = get_layer(options['encoder_general'])[1](tparams, reversed_emb_c_, options,
                                            prefix="%s_char_backward" % (options['encoder_general']),
                                            mask=reversed_m_,
                                            dim_proj_option_str='dim_char_proj')
        #return output
        combined =  tensor.concatenate([char_proj[-1], reversed_char_proj[-1]], axis=1)

        #combined =  tensor.concatenate([char_proj, reversed_char_proj], axis=1)

        #combined= combined * m_[:, :, None]
        #return combined[0]
        return combined

    oval, updates = theano.scan(_char_step,
                                sequences=[mask_char, emb_char],
                                name='lstm_char_embedding_layers',
                                n_steps=n_timesteps)

    emb_char_lstm = oval

    ## for character CNN


    n_context_oneside_val = int((options['n_char_context'] - 1)/ 2)

    mytaps = range(-n_context_oneside_val, n_context_oneside_val+1, 1)

    n_context_oneside = tensor.constant(numpy_intX(n_context_oneside_val))

    zero_emb = tensor.constant(numpy.zeros(options['dim_char_proj'], dtype=config.floatX))

    dummy = tensor.repeat(zero_emb, n_context_oneside * n_samples).reshape([n_context_oneside, n_samples, options['dim_char_proj']])

    def _convstep(*args):
        combined = tensor.concatenate(args, axis=1)

        return tensor.dot(combined, tparams['conv_W'])

    def _convstep_all(m_, emb_c_):
        emb_extended = tensor.concatenate([dummy,emb_c_,dummy])

        oval,updates = theano.scan(_convstep,
                sequences=dict(input=emb_extended,taps=mytaps),
                outputs_info=None)

        return tensor.max(oval, axis=0)

    oval,updates = theano.scan(_convstep_all,
                sequences=[mask_char, emb_char],
                name='cnn_char',
                n_steps=n_timesteps)

    emb_char_conv = oval

    # concatenate ...

    emb = tensor.concatenate([emb_word, emb_pos, emb_chunk, emb_char_lstm, emb_char_conv], axis=2)

    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)


    reversed_emb = emb[::-1]
    reversed_mask = mask[::-1]

    reversed_proj = get_layer(options['encoder'])[1](tparams, reversed_emb, options,
                                            prefix="%s_backward" % (options['encoder']),
                                            mask=reversed_mask)

    proj2 = reversed_proj[::-1]

    ##nsteps = proj.shape[0]
    # proj = n_timesteps * n_samples * n_proj

    proj = tensor.concatenate([proj, proj2], axis=2)

    if options['encoder'] == 'lstm':
        #proj = (proj * mask[:, :, None]).sum(axis=0)
        proj = proj * mask[:, :, None]


    #########################################################
    def _step(m_, h_):
        preact = tensor.dot(h_, tparams['U'])
        preact += tparams['b']

        ##output = tensor.nnet.softmax(preact)
        ##argmax =    output.argmax(axis=1)

        output = m_[:, None] * preact

        return output

    oval, updates = theano.scan(_step,
                                sequences=[mask, proj],
                                name='lstm_output_layers',
                                n_steps=n_timesteps)



    pred = oval


    if options['use_dropout']:
        pred = dropout_layer(pred, use_noise, trng)

    #
    starttagid = tensor.constant(numpy_intX(0))
    endtagid = tensor.constant(numpy_intX(1))


    initial_deltas      = tensor.repeat(tparams['A'][starttagid, :], n_samples).reshape((options['ydim'],n_samples)).dimshuffle(1,0)
    ##initial_deltas      =   tensor.log(initial_deltas)

    bestpathscores      = tensor.repeat(tparams['A'][starttagid, :], n_samples).reshape((options['ydim'],n_samples)).dimshuffle(1,0)
    bestprevstates      = tensor.repeat(starttagid, n_samples * options['ydim']).reshape((n_samples,options['ydim']))


    starttagids     =   tensor.repeat(starttagid, n_samples)

    yprevstates     =   starttagids


    def _crf_step(m_, obs_, y_, delta_, bscore_, bstate_, yprev_, ys_):
        # compute y
        yo_ = obs_[tensor.arange(n_samples), y_]
        ya_ = tparams['A'][yprev_, y_]

        yscores   =   yo_ + ya_ + ys_
        #yscores     =   m_[:, None] * yscores + (1-m_)[:, None] * ys_        #yscores =   ys_
        yscores     =   m_ * yscores + (1-m_) * ys_        #yscores =   ys_
        #ynewprev  =   tensor.cast(m_[:,None] * y_ + (1-m_)[:,None] * yprev_, 'int64')
        ynewprev  =   tensor.cast(m_ * y_ + (1-m_) * yprev_, 'int64')


        a_ = tparams['A'].dimshuffle('x', 1, 0)


        o_ = obs_.dimshuffle(0, 'x', 1)
        d_ = delta_.dimshuffle(0, 'x', 1)   #delta: n_samples * n_dim * n_dim

        #s_ = score_.dimshuffle(0, 1, 'x')   #bestscore
        s_ = bscore_.dimshuffle(0, 'x', 1)   #bestscore

        deltas  =  theano_logsumexp (o_ + d_ + a_, axis=2)

        preact = o_ + s_ + a_
        bestscores  = preact.max(axis=2)    #n_samples * n_dim
        beststates  = preact.argmax(axis=2) #n_samples * n_dim

        # masking
        deltas          =   m_[:, None] * deltas + (1-m_)[:, None] * delta_
        bestscores      =   m_[:, None] * bestscores + (1-m_)[:, None] * bscore_
        beststates      =   tensor.cast(m_[:, None] * beststates + (1-m_)[:, None] * bstate_, 'int64')


        return deltas, bestscores, beststates, ynewprev, yscores

    oval, updates = theano.scan(_crf_step,
                                sequences=[mask, pred, y],
                                outputs_info=[initial_deltas, bestpathscores, bestprevstates, yprevstates,
                                                    tensor.alloc(numpy_floatX(0.), n_samples)],
                                #non_sequences=transitionparams,
                                name='lstm_decode_layers',
                                n_steps=n_timesteps)

    #enda_  =    tensor.tile( tparams['A'][:, endtagid], [n_samples, 1]) ## n_samples, n_dims
    enda_      = tensor.repeat(tparams['A'][:, endtagid], n_samples).reshape((options['ydim'],n_samples)).dimshuffle(1,0)
    endtagids   =   tensor.repeat(endtagid, n_samples)

    ## oval[0][-1] ===> nsamples * n_dims
    deltas = oval[0][-1][:,endtagid]

    ynewprev = oval[3][-1]

    #oval[4][-1] ==> 1 * n_samples
    #yscores = oval[4][-1] + enda_[tensor.arange(n_samples), ynewprev]
    yscores = oval[4][-1] + tparams['A'][:, endtagid] [ynewprev]
    #yscores = oval[4][-1][endtagids]


    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    #cost = -(yscores[:,None] - deltas).mean()
    cost = -(yscores - deltas).mean()



    bestpathscores      = tensor.repeat(tparams['A'][starttagid, :], n_samples).reshape((options['ydim'],n_samples)).dimshuffle(1,0)
    bestprevstates      = tensor.repeat(starttagid, n_samples * options['ydim']).reshape((n_samples,options['ydim']))


    def _crf_predict_step(m_, obs_, score_, b_):
        ## o_.shape = (numsamples, ydim)
        # score.shape = (1, numsamples)
        #o_ = obs_.dimshuffle(0, 1, 'x')
        #s_ = score_.dimshuffle(0, 1, 'x')   #bestscore

        o_ = obs_.dimshuffle(0, 'x', 1)
        s_ = score_.dimshuffle(0, 'x', 1)   #bestscore
        a_ = tparams['A'].dimshuffle('x', 1, 0)

        preact = o_ + s_ + a_
        bestscores  = preact.max(axis=2)    #n_samples * n_dim
        beststates  = preact.argmax(axis=2) #n_samples * n_dim

        bestscores      =   m_[:, None] * bestscores + (1-m_)[:, None] * score_
        beststates      =   tensor.cast(m_[:, None] * beststates + (1-m_)[:, None] * b_, 'int64')

        #yscores  = preact[tensor.arange(n_timesteps).reshape((-1,1)), tensor.arange(n_samples), y_]
        return bestscores, beststates
        #return score_, b_

    pval, updates = theano.scan(_crf_predict_step,
                                sequences=[mask, pred],
                                outputs_info=[bestpathscores, bestprevstates],
                                #non_sequences=transitionparams,
                                name='lstm_predict_decode_layers',
                                n_steps=n_timesteps)


    bestscores      =   pval[0]     #n_timesteps * n_samples * n_dim
    beststates      =   pval[1]     #n_timesteps * n_samples * n_dim


    bestnewscores = pval[0][-1][:,endtagid]
    #bestnewstates = pval[1][-1][:,endtagid]

    bestnewstates   =   endtagids


    reversed_mask = mask[::-1]
    reversed_beststates = beststates[::-1]

    def _crf_backtrack_step(m_, b_, bn_):
        ## b_ : n_samples * n_dim
        newb =    b_[tensor.arange(n_samples), bn_]
        newb =     tensor.cast( m_ * newb + (1-m_) * bn_, 'int64')


        #return bestscores, beststates
        return newb

    bval, updates = theano.scan(_crf_backtrack_step,
                    sequences=[reversed_mask, reversed_beststates],
                    outputs_info=[bestnewstates],
                    name='lstm_predict_decode_layers',
                    n_steps=n_timesteps)


    beststates = bval[::-1]

    f_pred_prob = theano.function([x, x_pos, x_chunk, mask, x_char, mask_char, y], [yscores,deltas,bestnewscores,bestnewstates,pval[1],pred], name='f_pred_prob')

    f_pred = theano.function([x, x_pos, x_chunk, mask, x_char, mask_char], beststates, name='f_pred')

    return use_noise, x, x_pos, x_chunk, mask, x_char, mask_char, y, f_pred_prob, f_pred, cost



def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, x_char, mask_char, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)

        x_word = x[:,:,0]
        x_pos = x[:,:,1]
        x_chunk = x[:,:,2]

        pred_probs = f_pred_prob(x_word, x_pos, x_chunk, mask, x_char, mask_char)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print '%d/%d samples classified' % (n_done, n_samples)

    return probs


def pred_error(f_pred, prepare_data, data, iterator, verbose=False, valid_size=-1,  word_str_dict=None, tag_str_dict=None, predictto=None):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    num_valid = 0
    num_valid_timesteps = 0

    pred_results = []

    for _, valid_index in iterator:
        x, mask,  x_char, mask_char, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)

        x_word = x[:,:,0]
        x_pos = x[:,:,1]
        x_chunk = x[:,:,2]

        preds = f_pred(x_word, x_pos, x_chunk, mask, x_char, mask_char)

        #print preds
        targets = numpy.array(data[1])[valid_index]

        numsteps = x.shape[0]
        numsamples = x.shape[1]

        for i in range(numsteps):
            #valid_err += (preds[i] != targets[i]).sum()
            valid_err += ((preds[i] != y[i]) * mask[i]).sum()

            num_valid_timesteps += mask[i].sum()

        for i in range(numsamples):

            if predictto:
                localresults = []
                for j in range(len(preds[:,i])):
                    if mask[j,i] == 0.:
                        localresults = zip( [word_str_dict[v] for v in x_word[:j, i]] , [tag_str_dict[v] for v in preds[:j,i]] )
                        break

                if len(preds[:,i]) > 0 and len(localresults) == 0:
                    localresults = zip( [word_str_dict[v] for v in x_word[:,i]] , [tag_str_dict[v] for v in preds[:,i]] )

                pred_results.append(localresults)

        num_valid += 1

        if valid_size >= 0 and num_valid >= valid_size:
            break

    #valid_err = 1. - numpy_floatX(valid_err) / len(data[0])
    valid_err = numpy_floatX(valid_err) / numpy_floatX(num_valid_timesteps)

    if predictto:
        pkl.dump(pred_results, predictto, -1)

    return valid_err



def get_segment_result(tag_result, tag_segment_type_dict, tag_str_dict, mask):

    seg_result = {}

    start = -1
    start_tag_str = "null"

    end = len(tag_result) - 1
    for i in range(len(tag_result)):
        if mask[i] == 0.0:
            end = i - 1
            break

        type = tag_segment_type_dict[ tag_result[i] ]

        tag_str = tag_str_dict[tag_result[i]]

        if tag_str.startswith("O") is False:
            tag_str = tag_str[2:]

        if type == 0: ##B
            if start >= 0:
                seg_result[(start,i-1, start_tag_str)]  = 1
            start = i
            start_tag_str = tag_str
        elif type == 2: ##O
            if start >= 0:
                seg_result[(start,i-1, start_tag_str)]  = 1
            start = -1
            start_tag_str = "null"
        elif type == 1: ##I
            if start < 0:
                start = i
                start_tag_str = tag_str

    if start >= 0:
        seg_result[(start,end, start_tag_str)] = 1

    return seg_result


def segmentation_pred_error(f_pred, prepare_data, data, iterator, verbose=False, valid_size=-1, tag_segment_type_dict=None, word_str_dict=None, tag_str_dict=None, predictto=None):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    num_valid = 0
    num_valid_timesteps = 0

    num_total_answer = 0
    num_total_output = 0
    num_total_matched = 0

    pred_results = []

    for _, valid_index in iterator:
        x, mask, x_char, mask_char, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        x_word = x[:,:,0]
        x_pos = x[:,:,1]
        x_chunk = x[:,:,2]

        preds = f_pred(x_word, x_pos, x_chunk, mask, x_char, mask_char)

        #print preds
        ##targets = numpy.array(data[1])[valid_index]


        numsteps = x.shape[0]
        numsamples = x.shape[1]

        for i in range(numsamples):
            #valid_err += (preds[i] != targets[i]).sum()
            valid_err += ((preds[:,i] != y[:,i]) * mask[:,i]).sum()

            if predictto:
                localresults = []
                for j in range(len(preds[:,i])):
                    if mask[j, i] == 0.:
                        localresults = zip( [word_str_dict[v] for v in x_word[:j,i]] , [tag_str_dict[v] for v in preds[:j,i]] )
                        break

                if len(preds[:,i]) > 0 and len(localresults) == 0:
                    localresults = zip( [word_str_dict[v] for v in x_word[:,i]] , [tag_str_dict[v] for v in preds[:,i]] )

                pred_results.append(localresults)

            y_seg = get_segment_result(y[:,i], tag_segment_type_dict, tag_str_dict, mask[:,i])
            r_seg = get_segment_result(preds[:,i], tag_segment_type_dict, tag_str_dict, mask[:,i])


            num_answer = len(y_seg)
            num_output = len(r_seg)

            num_matched = 0

            for seg in y_seg:
                if r_seg.has_key(seg):
                    num_matched += 1

            num_total_answer += num_answer
            num_total_output += num_output
            num_total_matched   += num_matched

            num_valid_timesteps += mask[:,i].sum()

        num_valid += 1

        if valid_size >= 0 and num_valid >= valid_size:
            break

    pred = float(num_total_matched)/float(num_total_output)
    rec = float(num_total_matched)/float(num_total_answer)

    F1_score = 2.0*pred*rec / (pred + rec)

    #valid_err = 1. - numpy_floatX(valid_err) / len(data[0])
    valid_err = numpy_floatX(valid_err) / numpy_floatX(num_valid_timesteps)


    if predictto:
        pkl.dump(pred_results, predictto, -1)

    return valid_err, F1_score


#def test_pred_error(f_pred, f_pred_prob, prepare_data, data, iterator, verbose=False, valid_size=200):
def test_pred_error(f_pred, f_pred_prob, prepare_data, data, iterator, verbose=False, valid_size=-1):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    num_valid = 0
    num_valid_timesteps = 0

    for _, valid_index in iterator:
        x, mask, x_char, mask_char, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)

        x_word = x[:,:,0]
        x_pos = x[:,:,1]
        x_chunk = x[:,:,2]

        if 1 == 0:
            yscores,deltas,bscores,bstates,ballstates,pred = f_pred_prob(x,x_pos, x_chunk, mask, y)


            print "yscores", yscores
            print "deltas", deltas
            print "bscores", bscores
            print "pred[-1]", pred[-1]
            print "pred", pred


        if 1 == 0:
            for i in range(pred_probs.shape[0]):
                for j in range(pred_probs.shape[1]):
                    if mask[i][j] == 1:
                        print pred_probs[i][j]
                        print len(pred_probs[i][j]),  numpy.argmax(pred_probs[i][j]),
                                    #            print pred_probs[i][j].sum()
                    print


        preds = f_pred(x,x_pos, x_chunk, mask, x_char, mask_char )


        targets = numpy.array(data[1])[valid_index]


        numsteps = x.shape[0]
        numsamples = x.shape[0]

        for i in range(numsteps):

            #for j in range(preds.shape[1]):
            #    print "preds", preds[i][j]
            #    print "y", y[i][j]

            valid_err += ((preds[i] != y[i]) * mask[i]).sum()

            num_valid_timesteps += mask[i].sum()

        num_valid += 1

        if valid_size >= 0 and num_valid >= valid_size:
            break

    #valid_err = 1. - numpy_floatX(valid_err) / len(data[0])
    valid_err = numpy_floatX(valid_err) / numpy_floatX(num_valid_timesteps)

    return valid_err




def train_lstm(
    #dim_proj=128,  # word embeding dimension
    embedding_data = 'embeddings.pkl',
    n_char_context=5, # window size for ConvNet-based composition
    dim_proj=175,   # dim_word_proj + dim_postag_proj + dim_chunk_proj + dim_char_proj * 2
    dim_word_proj=50,  # word embeding dimension
    dim_postag_proj=25,
    dim_chunk_proj=25,
    dim_feat_proj=25,  # feat embeding dimension
    dim_char_proj=25, # char embedding dimension
    dim_conv_char_proj=25,
    dim_general_feat_proj=5,
    dim_lstm_hidden_proj=-1, #LSTM number of hidden units.
    #patience=10,  # Number of epoch to wait before early stop if no progress
    patience=100,  # Number of epoch to wait before early stop if no progress
    max_epochs=50,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    #n_words=10000,  # Vocabulary size
    n_words=100000,  # Vocabulary size
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    encoder_general='lstm_general',  # TODO: can be removed must be lstm.
    saveto='bilstmcrf_charlstmconv.npz',  # The best model will be saved there
    predictto=None,
    savetodir=None,
    #validFreq=370,  # Compute the validation error after this number of update.
    #validFreq=10,  # Compute the validation error after this number of update.
    #validFreq=370,  # Compute the validation error after this number of update.
    #validFreq=10,  # Compute the validation error after this number of update.
    #validFreq=100,  # Compute the validation error after this number of update.
    validFreq=300,
    #validFreq=370,  # Compute the validation error after this number of update.
    saveFreq=1110,  # Save the parameters after every saveFreq updates
    maxlen=100,  # Sequence longer then this get ignored
    batch_size=16,  # The batch size during training.
    #batch_size=4,  # The batch size during training.
    #batch_size=5,  # The batch size during training.
    #batch_size=1,  # The batch size during training.
    valid_batch_size=64,  # The batch size used for validation/test set.
    #valid_batch_size=4,  # The batch size used for validation/test set.
    dataset='penndb',

    # Parameter for extra option
    noise_std=0.,
    use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
    eval=False
):

    # Model options
    model_options = locals().copy()
    print "model options", model_options

    load_data, prepare_data, load_dict_data = get_dataset(dataset)

    print 'Loading data'
    print "n_words", n_words
    print "n_char_context", n_char_context

    if predictto:
        maxlen = None
        train, valid, test = load_data(n_words=n_words, valid_portion=0.05,
                        maxlen=maxlen, sort_by_len=False)

        print >>sys.stderr, "sort_by_len=False"
    else:
        train, valid, test = load_data(n_words=n_words, valid_portion=0.05,
                        maxlen=maxlen)


    #### seung-hoon

    #if savetodir:
    #    saveto = os.path.join(savetodir, saveto)
    #    print >>sys.stderr, "saveto: %s" % (saveto)

    if test_size > 0:
        # The test set is sorted by size, but we want to keep random
        # size example.  So we must select a random selection of the
        # examples.
        idx = numpy.arange(len(test[0]))
        numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])

    dicts =    load_dict_data()
    feat_dict = None
    char_dict = None

    n_feats = None
    n_chars = None

    word_dict, tag_dict, feat_dict, char_dict = dicts

    n_feats = []
    for my_feat_dict in feat_dict:
        n_feats.append(numpy.max(my_feat_dict.values()) + 1)

    n_feats = numpy.array(n_feats)

    word_str_dict = build_id2str_dict(word_dict)

    tag_str_dict = build_id2str_dict(tag_dict)

    tag_segment_type_dict = build_tag_segment_type_dict(tag_dict)

    #print train[1]
    print numpy.max(tag_dict.values()) + 1
    #print len(train[1][0]), len(train[1][1])
    #ydim = numpy.max(train[1]) + 1
    ydim = numpy.max(tag_dict.values()) + 1

    n_chars = numpy.max(char_dict.values()) + 1

    model_options['n_postags'] = n_feats[0]     # for postags
    model_options['n_chunks'] = n_feats[1]      # for chunks
    model_options['n_chars']  = n_chars

    print "n_postags", n_feats[0]
    print "n_chunks", n_feats[1]
    print "n_chars", n_chars

    dim_proj = dim_word_proj + dim_postag_proj + dim_chunk_proj + dim_char_proj * 2 + dim_conv_char_proj

    if dim_lstm_hidden_proj == None or dim_lstm_hidden_proj < 0:
        dim_lstm_hidden_proj = dim_proj
        model_options['dim_lstm_hidden_proj'] = dim_proj

    print "dim_proj", dim_proj
    print "dim_lstm_hidden_proj", dim_lstm_hidden_proj

    model_options['dim_proj'] = dim_proj


    #saveto = "bidirectional_lstm_model_full_dropout_directf_feat_general_cnn_char_pretrained_LSTM%d_%d_wordDim%d_charDim%d_ncontext%d.npz"  % (dim_proj, dim_lstm_hidden_proj, dim_word_proj, dim_char_proj, n_char_context)

    saveto = "bilstmcrf_charlstmconv%d_%d_wordDim%d_charDim%d_ncontext%d.npz"  % (dim_proj, dim_lstm_hidden_proj, dim_word_proj, dim_char_proj, n_char_context)


    #model_options['dim_proj'] = dim_proj
    if savetodir:
        saveto = os.path.join(savetodir, saveto)
        print >>sys.stderr, "saveto: %s" % (saveto)
    else:
        print >>sys.stderr, "saveto: %s" % (saveto)


    model_options['ydim'] = ydim

    print 'Building model'
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options, embeddings=embedding_data)

    if reload_model:
        #load1315_params('bidirectional_lstm_model.npz', params)
        load_params(saveto, params)

        print >> sys.stderr, "reload_model", saveto

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, x_pos, x_chunk, mask, x_char, mask_char,
     y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, x_pos, x_chunk, mask, x_char, mask_char, y], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=tparams.values())
    f_grad = theano.function([x, x_pos, x_chunk, mask, x_char, mask_char, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, x_pos, x_chunk, mask, x_char, mask_char, y, cost)

    print 'Optimization'

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print "%d train examples" % len(train[0])
    print "%d valid examples" % len(valid[0])
    print "%d test examples" % len(test[0])

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) / batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) / batch_size

    print "%s valid freq"  % (validFreq)
    print "%s save freq"  % (validFreq)

    if eval == False:
        uidx = 0  # the number of update done
        estop = False  # early stop
        start_time = time.time()
        try:
            for eidx in xrange(max_epochs):
                n_samples = 0

                # Get new shuffled index for the training set.
                kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

                for _, train_index in kf:
                    uidx += 1
                    use_noise.set_value(1.)

                    # Select the random examples for this minibatch
                    y = [train[1][t] for t in train_index]
                    x = [train[0][t]for t in train_index]

                    # Get the data in numpy.ndarray format
                    # This swap the axis!
                    # Return something of shape (minibatch maxlen, n samples)
                    x, mask, x_char, mask_char, y = prepare_data(x, y)

                    x_word = x[:,:,0]
                    x_pos =  x[:,:,1]
                    x_chunk =  x[:,:,2]

                    n_samples += x.shape[1]

                    cost = f_grad_shared(x_word, x_pos, x_chunk, mask, x_char, mask_char, y)
                    f_update(lrate)

                    if numpy.isnan(cost) or numpy.isinf(cost):
                        print 'NaN detected'
                        return 1., 1., 1.

                    if numpy.mod(uidx, dispFreq) == 0:
                        print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

                    if saveto and numpy.mod(uidx, saveFreq) == 0:
                        print 'Saving... to %s ' % (saveto),

                        if best_p is not None:
                            params = best_p
                        else:
                            params = unzip(tparams)
                        numpy.savez(saveto, history_errs=history_errs, **params)
                        pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                        print 'Done'

                    if numpy.mod(uidx, validFreq) == 0:
                        print 'Predict error...',
                        use_noise.set_value(0.)

                        #train_err = pred_error(f_pred, prepare_data, train, kf)
                        #valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
                        #test_err = pred_error(f_pred, prepare_data, test, kf_test)

                        #train_err = test_pred_error(f_pred, f_pred_prob, prepare_data, train, kf)
                        #valid_err = test_pred_error(f_pred, f_pred_prob, prepare_data, valid, kf_valid)
                        #test_err = test_pred_error(f_pred, f_pred_prob, prepare_data, test, kf_test)

                        train_err = 0
                        train_F1_score = 0

                        train_F1_score = 0
                        valid_F1_score = 0
                        test_F1_score = 0

                        valid_direct_err = 0
                        test_direct_err = 0

                        if has_segmentation_problem(tag_dict, dataset):
                            train_err, train_F1_score = segmentation_pred_error(f_pred, prepare_data, train, kf, tag_segment_type_dict=tag_segment_type_dict, tag_str_dict=tag_str_dict)
                            valid_err, valid_F1_score = segmentation_pred_error(f_pred, prepare_data, valid, kf_valid, tag_segment_type_dict=tag_segment_type_dict, tag_str_dict=tag_str_dict)
                            test_err, test_F1_score = segmentation_pred_error(f_pred, prepare_data, test, kf_test, tag_segment_type_dict=tag_segment_type_dict, tag_str_dict=tag_str_dict)

                            valid_direct_err = 1.0 - valid_F1_score
                            test_direct_err = 1.0 - test_F1_score

                        else:
                            train_err = pred_error(f_pred, prepare_data, train, kf)
                            valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
                            test_err = pred_error(f_pred, prepare_data, test, kf_test)


                            valid_direct_err = valid_err
                            test_direct_err = test_err

                        history_errs.append([valid_direct_err, test_direct_err])

                        if (uidx == 0 or
                            valid_direct_err <= numpy.array(history_errs)[:,
                                                                   0].min()):

                            best_p = unzip(tparams)
                            bad_counter = 0

                        print ('Train ', train_err, 'Valid ', valid_err,
                               'Test ', test_err)

                        print ('Train F1', train_F1_score, 'Valid F1', valid_F1_score,
                               'Test F1', test_F1_score)

                        if (len(history_errs) > patience and
                            valid_direct_err >= numpy.array(history_errs)[:-patience,
                                                                   0].min()):
                            bad_counter += 1
                            if bad_counter > patience:
                                print 'Early Stop!'
                                estop = True
                                break

                    #if 1 == 1:
                    #    estop = True
                    #    break

                print 'Seen %d samples' % n_samples

                if estop:
                    break

        except KeyboardInterrupt:
            print "Training interupted"

        end_time = time.time()
        if best_p is not None:
            zipp(best_p, tparams)
        else:
            best_p = unzip(tparams)

    else:
        print >>sys.stderr, "test only ..."

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)

    train_F1_score = 0
    valid_F1_score = 0
    test_F1_score = 0

    fpredict = None
    if predictto is not None:
        #if savetodir:
        #    predictto = os.path.join(savetodir, predictto)
        print >>sys.stderr, "predictto: %s" % (predictto)
        fpredict = open(predictto, 'wb')

    if has_segmentation_problem(tag_dict, dataset):
        train_err, train_F1_score = segmentation_pred_error(f_pred, prepare_data, train, kf_train_sorted, tag_segment_type_dict=tag_segment_type_dict, word_str_dict=word_str_dict, tag_str_dict=tag_str_dict, predictto=fpredict)
        valid_err, valid_F1_score = segmentation_pred_error(f_pred, prepare_data, valid, kf_valid, tag_segment_type_dict=tag_segment_type_dict, word_str_dict=word_str_dict, tag_str_dict=tag_str_dict, predictto=fpredict)
        test_err, test_F1_score = segmentation_pred_error(f_pred, prepare_data, test, kf_test, tag_segment_type_dict=tag_segment_type_dict, word_str_dict=word_str_dict, tag_str_dict=tag_str_dict, predictto=fpredict)

    else:
        train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted, word_str_dict=word_str_dict, tag_str_dict=tag_str_dict, predictto=fpredict)
        valid_err = pred_error(f_pred, prepare_data, valid, kf_valid,  word_str_dict=word_str_dict, tag_str_dict=tag_str_dict, predictto=fpredict)
        test_err = pred_error(f_pred, prepare_data, test, kf_test,  word_str_dict=word_str_dict, tag_str_dict=tag_str_dict, predictto=fpredict)


    if fpredict:
        fpredict.close()

    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err
    print 'Train F', train_F1_score, 'Valid F', valid_F1_score, 'Test F', test_F1_score

    if eval == False and saveto:
        numpy.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    train_F1_score=train_F1_score,valid_F1_score=valid_F1_score,
                    test_F1_score=test_F1_score, history_errs=history_errs, **best_p)

    if eval == False:
        print 'The code run for %d epochs, with %f sec/epochs' % (
            (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
        print >> sys.stderr, ('Training took %.1fs' %
                              (end_time - start_time))

    return train_err, valid_err, test_err, train_F1_score, valid_F1_score, test_F1_score


if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    dataset = "conll03featdb"
    savetodir = None
    eval = False
    predictto = None
    reload_model = False

    embedding_data = None
    n_char_context = 5
    dim_word_proj = 50
    dim_lstm_hidden_proj = None

    validfreq = 1100

    try:
        opts, args = getopt.getopt(sys.argv[1:], "c:f:t:ep:d:w:h:j:v", ["help", "param="])
    except getopt.GetoptError, err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        #usage()
        usage()
        sys.exit(2)

    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o == '-d':
            dataset = a
        elif o == '-f':
            validfreq = int(a)
        elif o == '-j':
            dim_word_proj = int(a)
            print >>sys.stderr, "dim_word_proj", dim_word_proj
        elif o == '-h':
            dim_lstm_hidden_proj = int(a)
            print >>sys.stderr, "dim_lstm_hidden_proj", dim_lstm_hidden_proj
        elif o == '-t':
            predictto = a
            eval = True
            print >>sys.stderr, "predictto:", a
        elif o == '-c':
            n_char_context = int(a)
        elif o == '-p':
            savetodir = a
            if os.path.exists(savetodir) == False:
                os.makedirs(savetodir)
            print >>sys.stderr, "savetodir: %s" % (savetodir)
        elif o == '-e':
            eval = True
        elif o == '-w':
            #Addition
            embedding_data = a
            if os.path.exists(embedding_data) == False:
                sys.exit("Not exists embedding_data")
        else:
            assert False, "unhandled option"

    if eval:
        reload_model = True

    train_lstm(
        max_epochs=50,
        n_char_context=n_char_context,
        dim_word_proj=dim_word_proj,
        dim_lstm_hidden_proj=dim_lstm_hidden_proj,
        ##test_size=500,
        ##test_size=-1,
        eval=eval,
        reload_model=reload_model,
        dataset=dataset,
        savetodir=savetodir,
        predictto=predictto,
        validFreq=validfreq,

        embedding_data=embedding_data
    )








