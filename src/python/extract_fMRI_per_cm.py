#!/usr/bin/env python
# coding: utf-8

import h5py
import numpy as np
from tqdm import trange
import os

DATAPATH = '../../data/'
WRITE_DIR = 'fMRI_resp_per_cm'

# fMRI信号の遅延時間 
SLIDE_LENGTH = 5

# データの格納ディレクトリを作成
os.makedirs(DATAPATH + WRITE_DIR, exist_ok=True)

train_cm_length = [30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,30,30,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,15,30,30,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,30,30,15,15,15,15,15,30,30,15,30,15,15,15,15,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30]

test_cm_length1 = [30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30]
test_cm_length2 = [30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30]

for r in trange(27, desc='resp'):
    RESP = '01' + str(r + 1).zfill(2)
    hdfpath = DATAPATH + 'resp1/' + RESP + '.hdf' if r < 14 else DATAPATH + 'resp2/' + RESP + '.hdf'
    newpath = DATAPATH + WRITE_DIR + '/' + RESP + '.hdf'
    
    with h5py.File(hdfpath,'r') as read_file:
        with h5py.File(newpath, 'w') as write_file:
            # train set
            l = SLIDE_LENGTH
            for i in trange(12, desc='EVSet', leave=False):
                for j in trange(30, desc='EVNo'):
                    TR_EVSet = str(i + 1).zfill(2)
                    TR_EVNo = str(j + 1).zfill(2)
                    
                    r = l + train_cm_length[i * 30 + j]
                    if ( r > 7200) :
                        write_file.create_dataset(TR_EVSet + '_' + TR_EVNo, 
                                                  data=np.vstack((read_file['resp_trn'][l:7200], read_file['resp_trn'][0: SLIDE_LENGTH])))
                    else :
                        write_file.create_dataset(TR_EVSet + '_' + TR_EVNo, data=read_file['resp_trn'][l:r])
                    l = r
            # test set 1
            l = SLIDE_LENGTH
            for j in trange(30, desc='EVNo'):
                TR_EVSet = '13'
                TR_EVNo = str(j + 1).zfill(2)
                
                r = l + test_cm_length1[j]
                if ( r > 600) :
                    write_file.create_dataset(TR_EVSet + '_' + TR_EVNo, 
                                              data=np.vstack((read_file['resp_val'][l:600], read_file['resp_val'][0: SLIDE_LENGTH])))
                else :
                    write_file.create_dataset(TR_EVSet + '_' + TR_EVNo, data=read_file['resp_val'][l:r])
                l = r
            # test set 2
            l = 600 + SLIDE_LENGTH
            for j in trange(30, desc='EVNo'):
                TR_EVSet = '14'
                TR_EVNo = str(j + 1).zfill(2)
                
                r = l + test_cm_length2[j]
                if ( r > 1200) :
                    write_file.create_dataset(TR_EVSet + '_' + TR_EVNo, 
                                              data=np.vstack((read_file['resp_val'][l:1200], read_file['resp_val'][600: 600 + SLIDE_LENGTH])))
                else :
                    write_file.create_dataset(TR_EVSet + '_' + TR_EVNo, data=read_file['resp_val'][l:r])
                l = r
