import sys
import os
import subprocess

import logging
logging.basicConfig(format='%(levelname)s %(asctime)s : %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RAW_DATA = 'raw_data/criteo_conversion_logs/data-1M.txt.gz'
PROC_DATA = 'data/criteo/data-1M.pkl'

def parse_ne(text):
    for line in text.splitlines():
        if line.find('NE:') >= 0:
            return float(line.split('NE:')[1])
    raise ValueError('Cannot find NE from ' + text)

if __name__ == '__main__':
    logger.info('Processing raw data')
    cmd = ['python', 'src/criteo/proc_raw_data.py', 
           RAW_DATA, PROC_DATA]
    logger.info('Executing: %s', cmd)
    subprocess.check_call(cmd)

    logger.info('\n\n\n\nLearning logistic regression')
    Cs = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 
          1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    NEs = []
    for C in Cs:
        cmd = ['python', 'src/criteo/learn_lr.py', PROC_DATA, '--C', str(C)]
        logger.info('Executing: %s', cmd)
        logs = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        NEs.append(parse_ne(logs))
        
    for C, NE in zip(Cs, NEs):
        logger.info('C = %g, NE = %g', C, NE)
