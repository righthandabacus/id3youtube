#!/usr/bin/python

import pickle
import requests
import unicodedata
from typing import List, Set, Tuple, Iterable

import bsddb3
import parsel

from lart import lart
from ltit import ltit

# Berkeley DB for memoization
BSDDB = 'memo.db'

class bdbwrapper:
    '''A wrapper to Berkeley DB, which the open and close of file is hidden
    Usage:
        Write:
            b = bsdwrapper(filename)
            b.put(key, value)
        Read:
            b = bsdwrapper(filename)
            cur = b.cursor()
            rec = cur.first()
            while rec:
                print(rec)  # (key, value)-tuple
                rec = cur.next()
        Look up:
            b = bsdwrapper(filename)
            value = b.get(key)  # None if key not found
        Delete:
            b = bsdwrapper(filename)
            b.delete(key)
    '''
    def __init__(self, dbfile):
        self.db = bsddb3.db.DB()
        # open file, create if not exists, and use hash index
        self.db.open(dbfile, None, bsddb3.db.DB_HASH, bsddb3.db.DB_CREATE)
    def __getattr__(self, name):
        return getattr(self.db, name)
    def __del__(self):
        self.db.close()

def memoize(db):
    '''memoization decorator to use Berkeley DB

    Args:
        db: Instance of bsdwrapper object
    '''
    def _deco(func):
        def _func(*arg, **kwargs):
            key = pickle.dumps([func.__qualname__, arg, kwargs])
            val = db.get(key)
            if val is None:
                val = func(*arg, **kwargs)
                db.put(key, pickle.dumps(val))
            else:
                val = pickle.loads(val)
            return val
        return _func
    return _deco

wrapper = bdbwrapper(BSDDB)

@memoize(wrapper)
def curl(url):
    '''Retrieve from a URL and return the content body
    '''
    return requests.get(url).text

def gen_urls() -> List[str]:
    '''Return urls as strings for the artist names from mojim. Example URLs:
        https://mojim.com/twzlha_01.htm
        https://mojim.com/twzlha_07.htm
        https://mojim.com/twzlhb_01.htm
        https://mojim.com/twzlhb_07.htm
        https://mojim.com/twzlhc_01.htm
        https://mojim.com/twzlhc_33.htm
    '''
    for a in ['a', 'b']:
        for n in range(1, 8):
            yield "https://mojim.com/twzlh{}_{:02d}.htm".format(a, n)
    for n in range(1, 34):
        yield "https://mojim.com/twzlhc_{:02d}.htm".format(n)

def _get_names() -> List[str]:
    '''Return names of artists from mojim'''
    for url in gen_urls():
        html = curl(url)
        selector = parsel.Selector(html)
        titles = selector.xpath("//ul[@class='s_listA']/li/a/@title").getall()
        for t in titles:
            name, _ = t.strip().rsplit(' ', 1)
            yield name

# @memoize(wrapper)
def get_names() -> List[str]:
    '''Return a long list of names'''
    return list(_get_names())

@memoize(wrapper)
def get_titles() -> List[str]:
    with open("titles.txt") as fp:
        lines = fp.readlines()
    return lines

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

def printvectors(vector):
    print("[")
    for tok in vector:
        print("\t{},".format(tok))
    print("],")

def main():
    names = set(get_names())
    titles = get_titles()
    feat = [features(title.strip(), names, lart, ltit) for title in titles]
    import pickle
    pickle.dump(feat, open("feat.pickle", "wb"))
        

if __name__ == '__main__':
    main()
