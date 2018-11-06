import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import time
import pdb
import pickle

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [0]
def tick():
    _iter[0] += 1

def plot(name, value):
    _since_last_flush[name][_iter[0]] = value

def flush(outf, logfile):
    prints = []

    for name, vals in _since_last_flush.items():
        prints.append("{}\t{}".format(name, np.mean([vals[key] for key in vals])))
        _since_beginning[name].update(vals)

        x_vals = np.sort([key for key in _since_beginning[name]])
        y_vals = [_since_beginning[name][x] for x in x_vals]

        plt.clf()
        plt.plot(x_vals, y_vals)
        plt.xlabel('iteration')
        plt.ylabel(name)
        plt.savefig(os.path.join(outf, name.replace(' ', '_')+'.jpg'))

    print("iter {}\t{}".format(_iter[0], "\t".join(prints)))
    with open(logfile,'a') as f:
        f.write("iter {}\t{}".format(_iter[0], "\t".join(prints)) + "\n")
    _since_last_flush.clear()
