#!/usr/bin/env python
# coding: utf-8

"""Train the classifier from predefined data in feat.pickle and write the classifer result into
mlp-trained.pickle
"""

import pickle

import pandas as pd
import numpy as np
#from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam

INPUT = 'feat.pickle'

def get_model(input_dim, activation="relu"):
    # Create MLP model with 3 output for 3 classes
    # Mimick scikit-learn MLPClassifier default: 3-layers, ReLU
    # optional set activation="sigmoid" for logistic function
    model = Sequential()
    model.add(Dense(100, input_dim=input_dim, activation=activation))
    model.add(Dense(3, activation="softmax"))
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def read_data():
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
    return incol, dframe

def main():
    incol, dframe = read_data()

    # Train and save
    # y: manual conversion to one-hot encoding
    label = ["a", "t", "x"]
    X = dframe[incol]
    y = np.array([[1 if x==y else 0 for y in label] for x in dframe['label']])
    print(X.shape)
    print(y.shape)
    clf = get_model(len(incol), "sigmoid")
    clf.fit(X, y, epochs=1000, batch_size=128)
    score = clf.evaluate(X, y, batch_size=128)
    print(dict(zip(clf.metrics_names, score)))
    clf.save("keras-trained.h5")
    with open("keras-trained.incol.pickle", "wb") as fp:
        pickle.dump([label, incol], fp)

def main2():
    from keras.models import load_model
    clf = load_model("keras-trained.h5")
    incol, dframe = read_data()
    incol = pickle.load(open("keras-trained.incol.pickle", "rb"))
    pred = clf.predict(dframe[incol])
    label = ["a", "t", "x"]
    pred = [label[np.argmax(x)] for x in pred]
    print(pred)

if __name__ == '__main__':
    main()
