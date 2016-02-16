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
    description="normalize raw text data to numpy format")
parser.add_argument('input_file', type=str)
parser.add_argument('--test_ratio', type=float, default=0.1)

if __name__ == '__main__':
    args = parser.parse_args()
    args.output_file = args.input_file + '.pkl'

    logger.info('Loading data from %s', args.input_file)
    data = pd.read_csv(args.input_file, sep='\t', header=None).as_matrix()
    logger.info('Loaded data of shape %s', data.shape)
    
    raw_user_id = data[:, 0]
    raw_item_id = data[:, 1]

    uuid, user_id = np.unique(raw_user_id, return_inverse=True)
    uiid, item_id = np.unique(raw_item_id, return_inverse=True)
    logger.info('Found %d unique users and %d unique items', 
                uuid.size, uiid.size)

    data[:, 0] = user_id
    data[:, 1] = item_id

    logger.info('Partitioning data to %g:%g', 
                1 - args.test_ratio, args.test_ratio)
    np.random.shuffle(data)
    cut = int(np.ceil(data.shape[0] * args.test_ratio))
    logger.info('Using %d examples for training and %d for testing', 
                data.shape[0] - cut, cut)
    
    data = {
        "uuid": uuid,
        "uiid": uiid,
        "rec_log_train": data[:cut, :],
        "rec_log_test": data[cut:, :]
    }

    logger.info('Saving processed data to %s', args.output_file)
    with open(args.output_file, 'wb') as out:
        pkl.dump(data, out, pkl.HIGHEST_PROTOCOL)
