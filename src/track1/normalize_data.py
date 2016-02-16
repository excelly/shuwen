import os, sys, time
import cPickle as pkl
import argparse

import numpy as np
import scipy as sp

import logging
logging.basicConfig(format='%(levelname)s %(asctime)s : %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(
    description="normalize raw text data to numpy format")
parser.add_argument('input_file', type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    args.output_file = args.input_file + '.pkl'

    logger.info('Loading data from %s', args.input_file)
    data = np.loadtxt(args.input_file, dtype=int, delimiter='\t')
    logger.info('Loaded data of shape %s', data.shape)
    
    raw_user_id = data[:, 0]
    raw_item_id = data[:, 1]

    uuid, user_id = np.unique(raw_user_id, return_inverse=True)
    uiid, item_id = np.unique(raw_item_id, return_inverse=True)
    logger.info('Found %d unique users and %d unique items', 
                uuid.size, uiid.size)

    data[:, 0] = user_id
    data[:, 1] = item_id
    data = {
        "uuid": uuid,
        "uiid": uiid,
        "rec_log": data
    }

    logger.info('Saving processed data to %s', args.output_file)
    with open(args.output_file, 'wb') as out:
        pkl.dump(data, out, pkl.HIGHEST_PROTOCOL)
