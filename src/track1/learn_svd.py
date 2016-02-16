import os, sys, time
import cPickle as pkl
import argparse

import numpy as np
import scipy as sp
import scipy.sparse as ssp
import sklearn as skl

import logging
logging.basicConfig(format='%(levelname)s %(asctime)s : %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def svd(M, k):
    U, s, Vt = ssp.linalg.svds(M, k=k)
    idx = np.argsort(s)[::-1]
    return U[:, idx], s[idx], Vt[idx, :]

def svd_reconstruct(U, s, Vt, k=None):
    if k is None: 
        return np.dot(U * s, Vt)
    else:
        return np.dot(U[:, :k] * s[:k], Vt[:k, :])

def sparse_rmse(Y, R):
    assert R.shape == Y.shape
    if not hasattr(R, 'row'):
        R = R.tocoo()
    return np.sqrt(np.power((R.data - Y[R.row, R.col]), 2).mean())

def sparse_set(Y, R):
    assert R.shape == Y.shape
    if not hasattr(R, 'row'):
        R = R.tocoo()
    Y[R.row, R.col] = R.data

parser = argparse.ArgumentParser(
    description='''
compute recommendations using svd imputation.

Reference: 
Spectral Regularization Algorithms for Learning Large Incomplete Matrices
''')
parser.add_argument('input_file', type=str)
parser.add_argument('--rank', type=int, default=30)
parser.add_argument('--lam', type=float, default=0)
parser.add_argument('--epoch', type=int, default=10)

if __name__ == '__main__':
    args = parser.parse_args()
    args.output_file = args.input_file + '.rec_svd'

    logger.info('Loading data from %s', args.input_file)
    with open(args.input_file, 'rb') as inFile:
        data = pkl.load(inFile)
    uuid = data['uuid']
    uiid = data['uiid']
    rec_log = data['rec_log_train']
    rec_log_test = data['rec_log_test']
    logger.info('Loaded %d users, %d items, and %d logs', 
                uuid.size, uiid.size, rec_log.shape[0])

    R_shape = (uuid.size, uiid.size)
    R_train = ssp.coo_matrix((rec_log[:, 2].astype(np.float32), 
                              (rec_log[:, 0], rec_log[:, 1])),
                             R_shape)
    R_test = ssp.coo_matrix((rec_log_test[:, 2].astype(np.float32), 
                             (rec_log_test[:, 0], rec_log_test[:, 1])),
                            R_shape)
    R_train.sum_duplicates()
    R_test.sum_duplicates()
    R_train.data = np.clip(R_train.data, -1, 1, R_train.data)
    R_test.data = np.clip(R_test.data, -1, 1, R_test.data)
    logger.info('Constructed rating matrix of shape %s, NNZ = %d, Sparsity = %g', 
                R_train.shape, R_train.nnz, 
                float(R_train.nnz) / R_train.shape[0] / R_train.shape[1])

    R = R_train.tocsr()
    for i in range(args.epoch):
        U, s, Vt = svd(R, args.rank)
        s = np.maximum(s - args.lam, 0)

        Y = svd_reconstruct(U, s, Vt)

        logger.info('Iter = %d. RMSE: Train = %0.4f, Test = %0.4f', 
                    i, sparse_rmse(Y, R_train), sparse_rmse(Y, R_test))

        sparse_set(Y, R_train)
        R = Y

    sys.exit(0)

    U, s, Vt = svd(R, args.rank)
    for k in range(args.rank):
        Y = svd_reconstruct(U, s, Vt, k)
        print k, sparse_rmse(Y, R_train)
        print k, sparse_rmse(Y, R_test)
