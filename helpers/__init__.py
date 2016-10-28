from __future__ import division, print_function, absolute_import
import cv2
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from helpers.functions import multiply_with_overflow, normalize, random_chunk2, Assay, AutoEncoder, create_assay, generate_assays

with open('textfiles/colourmap.txt','rb') as f:
    g = f.read().splitlines()
    C = np.array([[int(y) for y in x.split(' ')] for x in g])
    cm = mpl.colors.ListedColormap(C/255.0)

with open('textfiles/AssayLabels.csv') as f:
    labels = np.array([x.split(',') for x in f.read().splitlines()])


run_labels = labels[np.array([x[0] for x in labels]) == 'Run 2']
runIDs1 = ['{}_{}'.format(label[1], label[2] + '{:02d}'.format(int(label[3]))) for label in run_labels]
runIDs0 = np.unique(['_'.join(x.split('_')[0:2]) for x in os.listdir('data/Run01test/') if x.endswith('.tif')])

IDs1 = [runID for runID in runIDs1 if runID in runIDs0]
IDs0 = runIDs0

train = pickle.load(open('data/train.py', 'rb'))




