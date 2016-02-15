import os, sys, time
import argparse

import numpy as np
import scipy as sp

import logging
logging.basicConfig(format='%(levelname)s %(asctime)s : %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def procRow(line, user_interval, item_interval):
    if line[0] % user_interval == 0 and line[1] % item_interval == 0:
        return line
    else:
        return None

parser = argparse.ArgumentParser(
    description="process raw data to generate a cleaned, downsampled data set")
parser.add_argument('--input_file', type=str, 
                    default='raw_data/track1/rec_log_train.txt')
parser.add_argument('--output_file', type=str, 
                    default='data/track1/rec_log_train')
parser.add_argument('--line_sampling', type=float, default=0.1)
parser.add_argument('--user_sampling', type=float, default=1)
parser.add_argument('--item_sampling', type=float, default=1)

if __name__ == '__main__':
    args = parser.parse_args()

    line_interval = int(max(1, 1. / args.line_sampling))
    user_interval = int(max(1, 1. / args.user_sampling))
    item_interval = int(max(1, 1. / args.item_sampling))
    logger.info('Sampling lines with interval %d', line_interval)
    logger.info('Sampling users with interval %d', user_interval)
    logger.info('Sampling items with interval %d', item_interval)
    args.output_file = '{}.{}-{}-{}'.format(
        args.output_file, line_interval, user_interval, item_interval)
    logger.info('Output data to %s', args.output_file)

    with open(args.input_file) as in_file, \
         open(args.output_file, 'w') as out_file:
        st = time.time() - 1e-6
        numIn = 0
        numOut = 0
        for line in in_file:
            numIn += 1
            if numIn % 1000000 == 0:
                logger.info('NumIn = %d, NumOut = %d. QPS = %0.1f', 
                            numIn, numOut, numIn / (time.time() - st))
            if numIn % line_interval != 0: continue

            line = tuple(map(int, line.split('\t')[:3]))
            line = procRow(line, user_interval, item_interval)
            if line:
                numOut += 1
                out_file.write('%d\t%d\t%d\n' % line)
