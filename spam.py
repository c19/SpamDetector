#!/usr/bin/env python
# coding=utf-8
#    Responsible for extracte feature from raw text
#    C19<caoyijun2050@gmail.com>
import numpy
from collections import Counter
from sklearn.linear_model import LogisticRegression
from gaussian import Gaussian
from common import Vocabulary
from scipy.sparse import csr_matrix, vstack
import math
import re
import collections
import csv
import sys
from base import do, cache, unzip, ensure_unicode, breakpoint
import ipdb

class SpamDetector(object):
    logreg = LogisticRegression()
    words = Vocabulary()

    @classmethod
    def train(cls, chats):
        do(chats,
           list,
           unzip,
           lambda xy: (cls.construct_X(xy[0]), xy[1]),
           lambda xy: cls.logreg.fit(xy[0], xy[1]),
           )

    @classmethod
    def vectorize(cls, content):
        return do(cls.get_indices(content),
                  lambda indices: csr_matrix(([1] * len(indices), indices, [0, len(indices)]), dtype=numpy.float64,
                                             shape=(1, len(cls.words) + 1)).log1p(),
                  )

    @classmethod
    def construct_X(cls, contents):
        indptr = [0]
        indices = []
        for content in contents:
            content = ensure_unicode(content)
            indices.extend([cls.words[word] for word in content])
            indptr.append(len(indices))
        data = [1] * len(indices)
        cls.words.seal()
        return csr_matrix((data, indices, indptr), dtype=numpy.float64,
                          shape=(len(indptr) - 1, len(cls.words) + 1)).log1p()

    @classmethod
    def get_indices(cls, content):
        return do(ensure_unicode(content),
                  lambda c: [cls.words[word] for word in c],
                  )

    @classmethod
    def get_score(cls, content):
        return cls.logreg.decision_function(cls.vectorize(content))[0]

    @classmethod
    def add_chats(cls, chats):
        pass
