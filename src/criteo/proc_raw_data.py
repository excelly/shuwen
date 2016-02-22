import os, sys, time
import cPickle as pkl
import argparse

import numpy as np
import scipy as sp
import scipy.sparse as ssp
import pandas as pd

import logging
logging.basicConfig(format='%(levelname)s %(asctime)s : %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

NUM_FLOAT_FEATURES = 8
NUM_STRING_FEATURES = 9

def float_feat_name(i): return 'ff%d' % i
def str_feat_name(i): return 'sf%d' % i

parser = argparse.ArgumentParser(
    description="process raw data to generate a cleaned, downsampled data set")
parser.add_argument('input_file', type=str)
parser.add_argument('output_file', type=str)
parser.add_argument('--float_bin_size', type=float, default=10)
parser.add_argument('--sparse_bin_size', type=int, default=1000)

if __name__ == '__main__':
    args = parser.parse_args()

    df = pd.read_csv(
        args.input_file, sep='\t', header=None,
        names=['imp_time', 'conv_time'] + 
        [float_feat_name(i) for i in range(NUM_FLOAT_FEATURES)] + 
        [str_feat_name(i) for i in range(NUM_STRING_FEATURES)])

    logger.info('Loaded data with shape %s', df.shape)
    logger.info('Raw data:\n%s', df[:3])

    df['label'] = pd.notnull(df['conv_time']).astype(float)
    del df['imp_time']
    del df['conv_time']

    bin_percentiles = np.arange(
        0, 100 + args.float_bin_size, args.float_bin_size)
    for i in range(NUM_FLOAT_FEATURES):
        col = df[float_feat_name(i)]
        bins = np.percentile(col.dropna().as_matrix(), bin_percentiles)
        bins = np.unique(bins)
        df[float_feat_name(i)] = (pd.cut(col, bins, labels=False) + 1).fillna(0)

    for i in range(NUM_STRING_FEATURES):
        col = df[str_feat_name(i)]
        col = pd.Categorical(col)
        col = (col.codes + 1) % args.sparse_bin_size
        df[str_feat_name(i)] = col

    idx_start = 0
    for i in range(NUM_FLOAT_FEATURES):
        logger.info('Feature %s starts from index %d', 
                    float_feat_name(i), idx_start)
        df[float_feat_name(i)] = df[float_feat_name(i)] + idx_start
        idx_start = df[float_feat_name(i)].max() + 1

    for i in range(NUM_STRING_FEATURES):
        logger.info('Feature %s starts from index %d', 
                    str_feat_name(i), idx_start)
        df[str_feat_name(i)] = df[str_feat_name(i)] + idx_start
        idx_start = df[str_feat_name(i)].max() + 1

    logger.info('Encoded data:\n%s', df[:3])

    df = df.as_matrix()
    labels = df[:, -1]
    features = df[:, :-1]
    n, dim = features.shape

    I = np.tile(np.reshape(np.arange(n), (n, 1)), (1, dim))
    features = ssp.csr_matrix((np.ones(n * dim), (I.ravel(), features.ravel())))

    logger.info('Final data: Shape = %s, Sparsity = %g', 
                features.shape, 
                float(features.nnz) / (features.shape[0] * features.shape[1]))

    logger.info('Saving data to %s', args.output_file)
    with open(args.output_file, 'wb') as out:
        pkl.dump({
            "labels": labels,
            "features": features
        }, out, -1)
