#!/usr/bin/python

"""Deploy model of classifier: Takes MP3 file list from command line, run the filename through the classifer
engine to determine artist name and song title, if found, then add as ID3 tags

The feature extraction functions must be same as crawler-dbm.py as those are used to train the classifier
"""

import argparse
import fileinput
import functools
import os.path
import pickle
import re
import unicodedata
from typing import List, Set, Tuple, Iterable

import pandas as pd
import mutagen.id3

def condense(tagged: Iterable[Tuple[str, str]]) -> Iterable[Tuple[str, str]]:
    "Aggregate pairs of the same tag"
    tag, string = None, ''
    for t,s in tagged:
        if tag == t:
            string += s
        elif tag is None:
            tag, string = t, s
        else:
            yield (tag, string)
            tag, string = t, s
    if tag is not None:
        yield (tag, string)

def latincondense(tagged: Iterable[Tuple[str, str]]) -> Iterable[Tuple[str,str]]:
    "Aggregate Latin words (Lu then Ll) into one tuple and mangle the space (Zs) type"
    dash = ['-', '_', '\u2500', '\uff0d', ]
    tag, string = None, ''
    for t,s in tagged:
        if (tag, t) == ('Lu','Ll'):
            tag, string = 'Lul', string+s
        elif tag is None:
            tag, string = t, s
        else:
            if tag == 'Zs':
                yield (tag, ' ')
            elif tag in ['Pc','Pd'] and string in dash:
                yield ('Pd', '-')
            else:
                yield (tag, string)
            tag, string = t, s
    if tag is not None:
        yield (tag, string)

def strcondense(tagged: Iterable[Tuple[str,str]]) -> List[Tuple[str,str]]:
    "Condense latin string phases into one tuple, needs look ahead"
    strtype = ['Lu', 'Ll', 'Lul', 'Lstr']
    tagged = list(tagged)
    i = 1
    # Condense string into Lstr tag
    while i < len(tagged):
        # combine if possible, otherwise proceed to next
        if tagged[i][0] in strtype and tagged[i-1][0] in strtype:
            tagged[i-1:i+1] = [('Lstr', tagged[i-1][1]+tagged[i][1])]
        elif i>=2 and tagged[i][0] in strtype and tagged[i-1][0] == 'Zs' and tagged[i-2][0] in strtype:
            tagged[i-2:i+1] = [('Lstr', tagged[i-2][1]+' '+tagged[i][1])]
            i = i-1
        else:
            i += 1
    return tagged

def features(title: str, names: Set[str], lart, ltit):
    '''Convert a title string into features'''
    stopword = ['MV', 'Music Video', '歌詞', '高清', 'HD', 'Lyric Video', '版']
    quotes = ['()', '[]', '《》', '【】', '（）', "“”", "''", '""',  ] # Ps and Pe, Pi and Pf, also Po
    # condense string with tags
    tagstr = strcondense(latincondense(condense( [(unicodedata.category(c), c) for c in title] )))
    # add other features: within quote (for diff quotes), is before dash, is
    # after dash, strlen count, tok count, Lo tok count, str is in name set, str
    # is in stopword set
    qpos = {}
    strlen = 0
    Lo_cnt = 0
    vector = []
    # position offset features, forward and backward
    #   fzhtok, bzhtok: Ordinality of ideographic tokens, counting from forward and backward
    #   slen: strlen of the token
    #   titlen: sum of tokens' char length, i.e., excluding whitespaces
    #   flen, blen: sum of tokens' char length up to before the current token, counting from forward and backward
    #   ftok, btok: Ordinality of token in title, counting from forward and backward
    #   tag: token nature derived from unicode category name
    #   stopword: boolean, whether the token has stopword as substring
    #   name: boolean, whether the token is a member of the set of artist names
    #   dashbefore, dashafter: boolean, whether there is a dash-like token anywhere before or after
    #   various quotes: boolean, whether the token is inside that kind of quote
    for i, (t, s) in enumerate(tagstr):
        vector.append(dict(
            str=s, tag=t, ftok=i, flen=strlen, slen=len(s),
            stopword=any(w in s for w in stopword), name=(s in names)
        ))
        if t == 'Lo':
            Lo_cnt += 1
            vector[-1]['fzhtok'] = Lo_cnt
        strlen += len(s)
    for tok in vector:
        tok.update({
            'titlen': strlen,
            'btok': len(vector) - 1 - tok['ftok'],
            'blen': strlen - len(tok['str']) - tok['flen'],
        })
        if 'fzhtok' in tok:
            tok['bzhtok'] = Lo_cnt + 1 - tok['fzhtok']
    # bracket features
    for key in quotes + ['-']:
        qpos[key] = []
    for tok in vector:
        if tok['tag'] == 'Pd':
            qpos['-'].append(tok['ftok'])
        else:
            for quote in quotes:
                if tok['str'] in quote:
                    qpos[quote].append(tok['ftok'])
                    break
    for tok in vector:
        try:
            tok['dashbefore'] = tok['ftok'] > min(qpos['-'])
            tok['dashafter'] = tok['ftok'] < max(qpos['-'])
        except ValueError:
            pass
        for quote in quotes:
            if not qpos[quote]:
                continue
            inquote = [1 for i,j in zip(qpos[quote][::2], qpos[quote][1::2]) if i < tok['ftok'] < j]
            tok[quote] = bool(inquote)
    # manual label
    for tok in vector:
        if tok in lart:
            tok['label'] = 'a'
        elif tok in ltit:
            tok['label'] = 't'
    return vector

@functools.lru_cache()
def get_engine():
    incol, clf = pickle.load(open("mlp-trained.pickle", "rb"))
    return incol, clf

def add_id3(filename):
    '''Add id3 tags to a MP3 file, if possible'''
    basename = os.path.basename(filename)
    if not basename.lower().endswith(".mp3"):
        return # not MP3, do nothing
    # build features from title string
    title = re.sub("-.{10,15}$", "", basename[:-4])  # remove .mp3 suffix and the youtube id
    featvect = []
    for feat in features(title, [], [], []):
        feat['Lstr'] = feat['tag'].startswith('L')
        featvect.append(feat)
    dframe = pd.DataFrame(featvect).rename(columns={'《》':"angle", '（）':"paren", '【】':"square", '“”':"quote"})
    dframe['bracket'] = dframe.filter(['()', 'paren', "''", 'square', 'quote', '[]']).fillna(0).max(axis=1)
    # load classifier engine and classify
    incol, clf = get_engine()
    for col in incol:
        if col not in dframe.columns:
            dframe[col] = 0
    dframe[incol] = dframe[incol].fillna(0).astype('int')
    dframe['label'] = clf.predict(dframe[incol])
    # detect artist name and title
    artist = ' '.join(dframe[dframe['label'].eq('a')]['str'])
    title = ' '.join(dframe[dframe['label'].eq('t')]['str'])
    if artist and title:
        mp3 = mutagen.id3.ID3(filename)
        mp3.add(mutagen.id3.TPE1(encoding=3, text=artist))
        mp3.add(mutagen.id3.TIT2(encoding=3, text=title))
        mp3.save()
        print("Tagged %r: TPE1=%r TIT2=%r" % (filename, artist, title))

def main():
    parser = argparse.ArgumentParser(
                description="ID3 tagger",
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("mp3files", nargs="*", help="MP3 files to tag")
    parser.add_argument("-l", "--list", help="Read from file list")
    args = parser.parse_args()
    if args.list:
        args.mp3files += [line.strip("\n") for line in fileinput.input(args.list)]
    for path in args.mp3files:
        add_id3(path)

if __name__ == '__main__':
    main()
