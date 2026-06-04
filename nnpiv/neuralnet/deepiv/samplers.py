from __future__ import absolute_import, division, print_function, unicode_literals

import numpy
from keras import backend as K
from keras.engine.topology import InputLayer

if K.backend() == "theano":
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    _FLOATX = theano.config.floatX
else:
    import tensorflow as tf

def random_laplace(shape, mu=0., b=1.):
    """
    Draw random samples from a Laplace distriubtion.

    See: https://en.wikipedia.org/wiki/Laplace_distribution#Generating_random_variables_according_to_the_Laplace_distribution

    Parameters:
        shape (tuple): Output shape.
        mu (array-like): Mean parameter.
        b (array-like): Equality constraint vector.
    """
    U = K.random_uniform(shape, -0.5, 0.5)
    return mu - b * K.sign(U) * K.log(1 - 2 * K.abs(U))

def random_normal(shape, mean=0.0, std=1.0):
    """
    Random normal.

    Parameters:
        shape (tuple): Output shape.
        mean (object): Value for `mean`.
        std (array-like): Standard-deviation parameter.
    """
    return K.random_normal(shape, mean, std)

def random_multinomial(logits, seed=None):
    """
    Theano function for sampling from a multinomal with probability given by `logits`

    Parameters:
        logits (array-like): Unnormalized class probabilities.
        seed (int or None): Random seed.
    """
    if K.backend() == "theano":
        if seed is None:
            seed = numpy.random.randint(1, 10e6)
        rng = RandomStreams(seed=seed)
        return rng.multinomial(n=1, pvals=logits, ndim=None, dtype=_FLOATX)
    elif K.backend() == "tensorflow":
        return tf.one_hot(tf.squeeze(tf.compat.v1.multinomial(K.log(logits), num_samples=1)),
                          int(logits.shape[1]))

def random_gmm(pi, mu, sig):
    """
    Sample from a gaussian mixture model. Returns one sample for each row in
    the pi, mu and sig matrices... this is potentially wasteful (because you have to repeat
    the matrices n times if you want to get n samples), but makes it easy to implment
    code where the parameters vary as they are conditioned on different datapoints.

    Parameters:
        pi (array-like): Mixture probabilities.
        mu (array-like): Mean parameter.
        sig (array-like): Mixture standard deviations.
    """
    normals = random_normal(K.shape(mu), mu, sig)
    k = random_multinomial(pi)
    return K.sum(normals * k, axis=1, keepdims=True)


