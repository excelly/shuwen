import os, sys, time
import cPickle as pkl
import argparse

import numpy as np
import scipy as sp
import scipy.sparse as ssp

from sklearn.linear_model import LogisticRegression as LR
from sklearn import cross_validation as CV
import sklearn.metrics as metrics

import logging
logging.basicConfig(format='%(levelname)s %(asctime)s : %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(
    description="logistic regression")
parser.add_argument('input_file', type=str)
parser.add_argument('--C', type=float, default=100)
parser.add_argument('--cv_folds', type=int, default=3)

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.input_file, 'rb') as f:
        data = pkl.load(f)

    labels = data["labels"]
    features = data["features"]
    n, dim = features.shape
    prior = labels.sum() / labels.size
    logger.info('Loaded data of size %s. Prior = %0.3f', (n, dim), prior)

    classifier = LR(C=args.C, penalty='l2', solver='lbfgs')
    logger.info('Using classifier: \n%s', classifier)

    logger.info('Starting %d-fold cross validation', args.cv_folds)
    predictions = np.ones(n) * -1
    for fold, idx in enumerate(CV.KFold(n, n_folds=args.cv_folds, shuffle=True)):
        logger.info('Starting fold %d / %d', fold, args.cv_folds)
        trIdx, teIdx = idx
        classifier.fit(features[trIdx], labels[trIdx])
        predictions[teIdx] = classifier.predict_proba(features[teIdx])[:, 1]

    accuracy = metrics.accuracy_score(labels, predictions > 0.5)
    auc = metrics.roc_auc_score(labels, predictions)
    priorNLL = - prior * np.log(prior) - (1 - prior) * np.log(1 - prior)
    ne = metrics.log_loss(labels, predictions) / priorNLL
    cal = predictions.mean() / prior
    logger.info('Accuracy: %0.4f', accuracy)
    logger.info('AUC: %0.4f', auc)
    logger.info('NE: %0.4f', ne)
    logger.info('Calibration: %0.4f', cal)
