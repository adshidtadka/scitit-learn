#!/usr/bin/env python
# coding: utf-8

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import trange
import os

DATAPATH = '../../data/'

# データの格納ディレクトリを作成
os.makedirs(DATAPATH + 'pca', exist_ok=True)

for r in trange(27, desc='resp'):
    RESP = '01' + str(r + 1).zfill(2)
    hdfpath = DATAPATH + 'resp1/' + RESP + '.hdf' if r < 14 else DATAPATH + 'resp2/' + RESP + '.hdf'
    pcapath = DATAPATH + 'pca/' + RESP + '.hdf'
    with h5py.File(hdfpath,'r') as read_file:
        with h5py.File(pcapath, 'w') as write_file:
            for data_name in ['resp_trn', 'resp_val']:
                pca = PCA(n_components=1000)
                pca.fit(read_file[data_name])
                write_file.create_dataset(data_name, data=pca.fit_transform(read_file[data_name]))

