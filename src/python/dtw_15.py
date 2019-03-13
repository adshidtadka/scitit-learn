import multiprocessing

import h5py
import numpy as np
from dtw import accelerated_dtw
from tqdm import tqdm

# %%

DATAPATH  = '../../brain_p/data/pca_fMRI_per_cm'
WRITEPATH = '../../brain_p/data/dtw15'

# 30秒と15秒を分ける
train_cm_length = [30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,30,30,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,15,30,30,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,30,30,15,15,15,15,15,30,30,15,30,15,15,15,15,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30]
test_cm_length1 = [30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30]
test_cm_length2 = [30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30,15,15,15,15,15,30,30]
cm_length = []
cm_length.extend(train_cm_length)
cm_length.extend(test_cm_length1)
cm_length.extend(test_cm_length2)

fifteen_index = []
for i in range(420):
    if (cm_length[i] == 15):
        fifteen_index.append(i)

# %% 連番からevset, evnoをつくる


def make_ev(num):
    """連番からevset, evnoをつくる."""
    counter = 0
    while((num + 1) % 30 != 0):
        num += 1
        counter += 1
    evset = int((num + 1) / 30)
    evno = int(30 - counter)
    return str(evset).zfill(2) + "_" + str(evno).zfill(2)


# %%


N_P = 27  # 被験者数
N_C = 280  # コンテンツ数


def eval_pair(pair):
    """ある被験者の動画aと動画bの類似度を計算する."""
    person, a, b = pair
    hdfpath = DATAPATH + "/01" + str(person + 1).zfill(2) + ".hdf"
    with h5py.File(hdfpath, 'r') as read_file:
        x = read_file[make_ev(a)].value
        y = read_file[make_ev(b)].value
        dist, cost, acc, path = accelerated_dtw(x, y, 'euclidean')
        return dist


def eval_person(person):
    """ある被験者のコンテンツ類似度の行列を作成する."""
    print('eval_person: {:2d}'.format(person))

    pairs = []
    for i in range(N_C):
        for j in range(i + 1, N_C):
            pairs.append((person, fifteen_index[i], fifteen_index[j]))

    similarity_between_pairs = None
    with multiprocessing.Pool() as p:
        similarity_between_pairs = p.map(eval_pair, tqdm(pairs))

    adj_mat = np.zeros((N_C, N_C))
    k = 0
    for i in range(N_C):
        for j in range(i + 1, N_C):
            adj_mat[i, j] = adj_mat[j, i] = similarity_between_pairs[k]
            k += 1

    writepath = WRITEPATH + "/01" + str(person + 1).zfill(2) + ".hdf"
    with h5py.File(writepath, 'w') as outfile:
        outfile.create_dataset('adjacency_matrix', data=adj_mat)


for person in range(N_P):
    eval_person(person)
