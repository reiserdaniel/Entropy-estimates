import numpy as np
from scipy.special import digamma
from scipy.special import gamma
from sklearn.neighbors import KDTree


#############################################################################################
#
# Itt gyujtom a forrasok/linkek listajat.
#
# entropy wiki:
#     https://en.wikipedia.org/wiki/Differential_entropy
#     https://en.wikipedia.org/wiki/Volume_of_an_n-ball
#     https://en.wikipedia.org/wiki/Lp_space
#     https://en.wikipedia.org/wiki/Particular_values_of_the_gamma_function
#
# numpy guide:
#     https://docs.scipy.org/doc/numpy-1.16.1/numpy-user-1.16.1.pdf
#     https://docs.scipy.org/doc/numpy/reference/generated/numpy.full_like.html#numpy.full_like
#
# random guide:
#     https://docs.python.org/3/library/random.html
#
# scikit forrasok:
#     https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html
#     https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
#     https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neighbors/binary_tree.pxi
#
# scipy forrasok (nem mind hasznalt):
#     https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.digamma.html
#     https://github.com/scipy/scipy/blob/master/scipy/spatial/kdtree.py
#
#
############################################################################################

def kraskov_entropy(x, k_range, p=1):
    # data is n*m matrix, where n is the number of sample elements, m is the attributes
    n = x.shape[0]
    d = x.shape[1]

    if k_range[-1] > n:
        raise Exception('k may not be longer than the data array')

    ks_dist, k2s_dist = __get_k_arrays(x, k_range, p)

    component_digamma = np.subtract(
        np.full(shape=len(k_range),
                fill_value=digamma(n)),
        [digamma(r) for r in k_range])

    if p is np.inf:
        component_cd = 0

    else:
        component_cd = np.full(
            np.shape(
                k_range),
            np.log2(
                np.divide(
                    np.power(
                        np.multiply(
                            2.,
                            gamma(1. / p + 1.)),
                        d),
                    gamma((d / p) + 1.))))

    component_rk = np.sum(
        np.multiply(
            np.full_like(
                ks_dist,
                x.shape[1] / x.shape[0]),
            np.log2(ks_dist)),
        axis=0)

    k_values = component_digamma \
        + component_cd \
        + component_rk

    k_mean = np.sum(k_values) / np.size(k_range, axis=0)

    return k_mean, k_values


def entropy_estimate_v1(x, k_range, p=1):
    # data is n*m matrix, where n is the number of sample elements, m is the attributes
    n = x.shape[0]
    d = x.shape[1]

    if k_range[-1] > n:
        raise Exception('k may not be longer than the data array')

    ks_dist, k2s_dist = __get_k_arrays(x, k_range, p)

    component_digamma = np.subtract(
        np.full(shape=len(k_range),
                fill_value=digamma(n)),
        [digamma(r) for r in k_range])

    if p is np.inf:
        component_cd = 0

    else:
        component_cd = np.full(
            np.shape(
                k_range),
            np.log2(
                np.divide(
                    np.power(
                        np.multiply(
                            2.,
                            gamma(1. / p + 1.)),
                        d),
                    gamma((d / p) + 1.))))

    component_dx = np.sum(
        np.divide(
            np.full_like(
                ks_dist,
                1. / n),
            np.log2(
                np.divide(
                    k2s_dist,
                    ks_dist))),
        axis=0)

    component_rk = np.sum(
        np.multiply(
            np.full_like(
                ks_dist,
                1. / n),
            np.log2(ks_dist)),
        axis=0)

    k_values = component_digamma \
        + component_cd \
        + np.multiply(component_dx, component_rk)

    k_mean = np.sum(k_values) / np.size(k_range, axis=0)

    return k_mean, k_values


def entropy_estimate_v2(x, k_range, p=1):
    # data is n*m matrix, where n is the number of sample elements, m is the attributes
    n = x.shape[0]
    d = x.shape[1]

    if k_range[-1] > n:
        raise Exception('k may not be longer than the data array')

    [ks_dist, k2s_dist] = __get_k_arrays(x, k_range, p)

    component_digamma = np.subtract(
        np.full(
            shape=len(k_range),
            fill_value=digamma(n)),
        [digamma(r) for r in k_range])

    if p is np.inf:
        component_cd = 0

    else:
        component_cd = np.full(
            np.shape(
                k_range),
            np.log2(
                np.divide(
                    np.power(
                        np.multiply(
                            2.,
                            gamma(1. / p + 1.)),
                        d),
                    gamma((d / p) + 1.))))

    component_dx = np.sum(
        np.multiply(
            np.divide(
                np.full_like(
                    ks_dist,
                    1. / n),
                np.log2(
                    np.divide(
                        k2s_dist,
                        ks_dist))),
            np.log2(ks_dist)),
        axis=0)

    k_values = component_digamma \
        + component_dx \
        + component_cd
    k_mean = np.sum(k_values) / np.size(k_range, axis=0)

    return k_mean, k_values


def __get_k_arrays(x, k_range, p):
    # using scipy for this task is also reasonable, but sklearn felt faster
    # it would be something like:
    #     from scipy.spatial import cKDTree
    #     tree = KDTree(data)
    #     distance, idndex = tree.query(numbers, k_max, p=mp.inf)
    data_np = np.array(x)

    # more metrics at https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
    if p is 1:
        metric = 'manhattan'
    elif p is 2:
        metric = 'euclidean'
    elif p is np.inf:
        metric = 'chebyshev'
    else:
        metric = 'euclidean'

    dist_tree = KDTree(data_np, leaf_size=2 * k_range[-1], metric=metric)
    dist_array, index = dist_tree.query(data_np, k=k_range[-1] * 2 + 1)

    ks_dist = np.array(dist_array[:, k_range])
    k2s_dist = np.array(dist_array[:, [r * 2 for r in k_range]])
    return ks_dist, k2s_dist
