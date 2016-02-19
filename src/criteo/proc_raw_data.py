import os, sys, time
import cPickle as pkl
import argparse

import numpy as np
import scipy as sp
import pandas as pd

import logging
logging.basicConfig(format='%(levelname)s %(asctime)s : %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(
    description="process raw data to generate a cleaned, downsampled data set")
parser.add_argument('input_file', type=str)
parser.add_argument('output_file', type=str)
parser.add_argument('--float_bin_size', type=float, default=10)
parser.add_argument('--sparse_bin_size', type=int, default=1000)

if __name__ == '__main__':
    args = parser.parse_args()

    def ff(i): return 'ff%d' % i
    def cf(i): return 'cf%d' % i

    df = pd.read_csv(args.input_file, sep='\t', header=None,
                       names=['imp_time', 'conv_time'] + 
                       [ff(i) for i in range(8)] + 
                       [cf(i) for i in range(9)])
    logger.info('Loaded data with shape %s', df.shape)

    print df[:5]
    print ''

    df['label'] = pd.notnull(df['conv_time']).astype(float)
    del df['imp_time']
    del df['conv_time']

    bin_percentiles = np.arange(0, 100 + args.float_bin_size, args.float_bin_size)
    for i in range(i):
        col = df[ff(i)]
        bins = np.unique(np.percentile(col.dropna().as_matrix(), 
                                       bin_percentiles))
        df[ff(i)] = (pd.cut(col, bins, labels=False) + 1).fillna(0)

    for i in range(9):
        col = df[cf(i)]
        col = pd.Categorical(col)
        col = (col.labels + 1) % args.sparse_bin_size
        df[cf(i)] = col + 1

    print df[:5]
