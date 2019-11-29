#!/usr/bin/env python
# coding: utf-8

"""Train the classifier from predefined data in feat.pickle and write the classifer result into
mlp-trained.pickle
"""

import pickle

import pandas as pd
from sklearn.neural_network import MLPClassifier

INPUT = 'feat.pickle'

def main():
    # Read from the pre-generated data and create feature vectors in dataframe
    # feat: list of list of dict, each list of dict is for one headline
    allfeat = [y for x in pickle.load(open(INPUT, "rb")) for y in x]

    # vectorize features
    featvect = []
    for feat in allfeat:
        feat = dict(feat)  # copy the dict
        feat['label'] = feat.get('label') or 'x'  # a, t, or x
        feat['Lstr'] = feat['tag'].startswith('L')
        featvect.append(feat)

    # convert into pandas dataframe (converted most to int)
    # resulting column set:
    #    ['ftok', 'flen', 'slen', 'stopword', 'name', 'titlen', 'btok', 'blen', 'dashbefore', 'dashafter',
    #     '[]', 'Lstr', 'fzhtok', 'bzhtok', '()', 'angle', "''", 'paren', 'square', 'quote', 'bracket']
    # non-int cols: str, label, tag are strings
    dframe = pd.DataFrame(featvect).rename(columns={'《》':"angle", '（）':"paren", '【】':"square", '“”':"quote"})
    dframe['bracket'] = dframe[['()', 'paren', "''", 'square', 'quote', '[]']].fillna(0).max(axis=1)

    # normalize data type
    incol = ['titlen', 'ftok', 'btok', 'flen', 'blen', 'slen', 'fzhtok', 'bzhtok',
             'stopword', 'dashbefore', 'dashafter', 'bracket', 'Lstr', 'angle']
    dframe[incol] = dframe[incol].fillna(0).astype('int')
    numcols = ['titlen', 'ftok', 'btok', 'flen', 'blen', 'slen', 'fzhtok', 'bzhtok']  # non-boolean
    incol = [c for c in incol if c in numcols] + sorted([c for c in incol if c not in numcols])

    # Train and save
    X = dframe[incol]
    y = dframe['label']
    clf = MLPClassifier(alpha=0.01, max_iter=1000, activation='logistic')
    clf.fit(X, y)
    with open("mlp-trained.pickle", "wb") as fp:
        pickle.dump([incol, clf], fp)

if __name__ == '__main__':
    main()
