# CS114B Spring 2023 Homework 4
# Part-of-speech Tagging with Structured Perceptrons

import os
import numpy as np
from collections import defaultdict
from random import Random

class POSTagger():

    def __init__(self):
        # for testing with the toy corpus from the lab 7 exercise
        self.tag_dict = {'nn': 0, 'vb': 1, 'dt': 2}
        self.word_dict = {'Alice': 0, 'admired': 1, 'Dorothy': 2, 'every': 3,
                          'dwarf': 4, 'cheered': 5}
        self.initial = np.array([-0.3, -0.7, 0.3])
        self.transition = np.array([[-0.7, 0.3, -0.3],
                                    [-0.3, -0.7, 0.3],
                                    [0.3, -0.3, -0.7]])
        self.emission = np.array([[-0.3, -0.7, 0.3],
                                  [0.3, -0.3, -0.7],
                                  [-0.3, 0.3, -0.7],
                                  [-0.7, -0.3, 0.3],
                                  [0.3, -0.7, -0.3],
                                  [-0.7, 0.3, -0.3]])
        # should raise an IndexError; if you come across an unknown word, you
        # should treat the emission scores for that word as 0
        self.unk_index = np.inf

    '''
    Fills in self.tag_dict and self.word_dict, based on the training data.
    '''
    def make_dicts(self, train_set):
        # iterate over training documents
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    pass

    '''
    Loads a dataset. Specifically, returns a list of sentence_ids, and
    dictionaries of tag_lists and word_lists such that:
    tag_lists[sentence_id] = list of part-of-speech tags in the sentence
    word_lists[sentence_id] = list of words in the sentence
    '''
    def load_data(self, data_set):
        sentence_ids = []
        tag_lists = dict()
        word_lists = dict()
        # iterate over documents
        for root, dirs, files in os.walk(data_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # be sure to split documents into sentences here
                    pass
        return sentence_ids, tag_lists, word_lists

    '''
    Implements the Viterbi algorithm.
    Use v and backpointer to find the best_path.
    '''
    def viterbi(self, sentence):
        T = len(sentence)
        N = len(self.tag_dict)
        v = np.zeros((N, T))
        backpointer = np.zeros((N, T), dtype=int)
        # initialization step
        # recursion step
        # termination step
        best_path = []
        return best_path

    '''
    Trains a structured perceptron part-of-speech tagger on a training set.
    '''
    def train(self, train_set):
        self.make_dicts(train_set)
        sentence_ids, tag_lists, word_lists = self.load_data(train_set)
        Random(0).shuffle(sentence_ids)
        self.initial = np.zeros(len(self.tag_dict))
        self.transition = np.zeros((len(self.tag_dict), len(self.tag_dict)))
        self.emission = np.zeros((len(self.word_dict), len(self.tag_dict)))
        for i, sentence_id in enumerate(sentence_ids):
            # your code here
            if (i + 1) % 1000 == 0 or i + 1 == len(sentence_ids):
                print(i + 1, 'training sentences tagged')

    '''
    Tests the tagger on a development or test set.
    Returns a dictionary of sentence_ids mapped to their correct and predicted
    sequences of part-of-speech tags such that:
    results[sentence_id]['correct'] = correct sequence of tags
    results[sentence_id]['predicted'] = predicted sequence of tags
    '''
    def test(self, dev_set):
        results = defaultdict(dict)
        sentence_ids, tag_lists, word_lists = self.load_data(dev_set)
        for i, sentence_id in enumerate(sentence_ids):
            # your code here
            if (i + 1) % 1000 == 0 or i + 1 == len(sentence_ids):
                print(i + 1, 'testing sentences tagged')
        return results

    '''
    Given results, calculates overall accuracy.
    '''
    def evaluate(self, results):
        accuracy = 0.0
        return accuracy

if __name__ == '__main__':
    pos = POSTagger()
    # make sure these point to the right directories
    #pos.train('brown/train')
    pos.train('data_small/train')
    #results = pos.test('brown/dev')
    results = pos.test('data_small/test')
    print('Accuracy:', pos.evaluate(results))
