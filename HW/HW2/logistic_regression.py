# CS114B Spring 2023 Homework 2
# Logistic Regression Classifier

import os
import numpy as np
from collections import defaultdict
from math import ceil
from random import Random
from scipy.special import expit # logistic (sigmoid) function

class LogisticRegression():

    def __init__(self):
        self.class_dict = {}
        # use of self.feature_dict is optional for this assignment
        self.feature_dict = {}
        self.n_features = None
        self.theta = None # weights (and bias)

    '''
    Given a training set, fills in self.class_dict (and optionally,
    self.feature_dict), as in HW1.
    Also sets the number of features self.n_features and initializes the
    parameter vector self.theta.
    '''
    def make_dicts(self, train_set):
        # iterate over training documents
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    pass
        # fill in class_dict, (feature_dict,) n_features, and theta
        # the following are used for testing with the toy corpus from the lab 3
        # exercise
        self.class_dict = {'action': 0, 'comedy': 1}
        self.feature_dict = {'fast': 0, 'couple': 1, 'shoot': 2, 'fly': 3}
        self.n_features = 4
        self.theta = np.zeros(self.n_features + 1)

    '''
    Loads a dataset. Specifically, returns a list of filenames, and dictionaries
    of classes and documents such that:
    classes[filename] = class of the document
    documents[filename] = feature vector for the document (use self.featurize)
    '''
    def load_data(self, data_set):
        filenames = []
        classes = dict()
        documents = dict()
        # iterate over documents
        for root, dirs, files in os.walk(data_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    pass
        return filenames, classes, documents

    '''
    Given a document (as a list of words), returns a feature vector.
    Note that the last element of the vector, corresponding to the bias, is a
    "dummy feature" with value 1.
    '''
    def featurize(self, document):
        vector = np.zeros(self.n_features + 1)
        # your code here
        vector[-1] = 1
        return vector

    '''
    Trains a logistic regression classifier on a training set.
    '''
    def train(self, train_set, batch_size=3, n_epochs=1, eta=0.1):
        filenames, classes, documents = self.load_data(train_set)
        filenames = sorted(filenames)
        n_minibatches = ceil(len(filenames) / batch_size)
        for epoch in range(n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
            loss = 0
            for i in range(n_minibatches):
                # list of filenames in minibatch
                minibatch = filenames[i * batch_size: (i + 1) * batch_size]
                # create and fill in matrix x and vector y
                # compute y_hat
                # update loss
                # compute gradient
                # update weights (and bias)
            loss /= len(filenames)
            print("Average Train Loss: {}".format(loss))
            # randomize order
            Random(epoch).shuffle(filenames)

    '''
    Tests the classifier on a development or test set.
    Returns a dictionary of filenames mapped to their correct and predicted
    classes such that:
    results[filename]['correct'] = correct class
    results[filename]['predicted'] = predicted class
    '''
    def test(self, dev_set):
        results = defaultdict(dict)
        filenames, classes, documents = self.load_data(dev_set)
        for name in filenames:
            # get most likely class (recall that P(y=1|x) = y_hat)
            pass
        return results

    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''
    def evaluate(self, results):
        # you can copy and paste your code from HW1 here
        pass

if __name__ == '__main__':
    lr = LogisticRegression()
    # make sure these point to the right directories
    #lr.make_dicts('movie_reviews/train')
    lr.make_dicts('movie_reviews_small/train')
    #lr.train('movie_reviews/train', batch_size=3, n_epochs=1, eta=0.1)
    lr.train('movie_reviews_small/train', batch_size=3, n_epochs=1, eta=0.1)
    #results = lr.test('movie_reviews/dev')
    results = lr.test('movie_reviews_small/test')
    lr.evaluate(results)
