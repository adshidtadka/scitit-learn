
# coding: utf-8

# In[357]:


import h5py
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series

# In[403]:


DATAPATH = 'data'


def get_xy(SEC, N_CLUSTER, METRIC):

    distancepath = DATAPATH + '/usermap{}.hdf'.format(SEC)
    with h5py.File(distancepath, 'r') as distance_file:
        # sample distance matrix の名残り
        sdm = distance_file['adjacency_matrix'].value
        if SEC == 15:
            sdm = sdm / (240 * 240)
        elif SEC == 30:
            sdm = sdm / (120 * 120)
        else:
            raise Error('invalid SEC: {}'.format(SEC))

    # sample clustering matrix
    scm = np.load(file=DATAPATH
                  + "/cluster/cluster_{}_{:02d}.npy".format(METRIC, N_CLUSTER))

    # sample flat list from sdm
    sfldm = []
    for i in range(27):
        for j in range(i + 1, 27):
            sfldm.append(sdm[i, j])

    # sample flat list from scm
    sflcm = []
    for i in range(27):
        for j in range(i + 1, 27):
            sflcm.append(scm[i, j])

    # 01_16みたいなkeyを与える
    keys = []
    for i in range(27):
        for j in range(i + 1, 27):
            keys.append(str(i + 1).zfill(2) + "_" + str(j + 1).zfill(2))

    dmdic = dict(zip(keys, sfldm))
    cmdic = dict(zip(keys, sflcm))

    # ソート
    dmser = Series(dmdic)
    dmser = dmser.sort_values()
    print(dmser.keys()[:50])

    sorted_cmlist = []
    sorted_keylist = []
    for i in dmser.keys():
        if (not ("16" in i)):
            sorted_cmlist.append(cmdic[i])
            sorted_keylist.append(i)

    dmser[sorted_keylist]

    # 積算値を計算
    numerator = 0
    denominator = 0
    value_list = []
    for i in sorted_cmlist:
        denominator += 1
        if (i == 1):
            numerator += 1
        value_list.append(numerator / denominator)
    print(len(value_list))

    x = dmser[sorted_keylist].values
    y = value_list

    return x, y


metrics = ['ham', 'jac']
metric_names = {
    'ham': 'Hamming',
    'jac': 'Jaccard',
}
n_cluster_list = {
    'ham': [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 20, 23, 25],
    'jac': [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21,
        22, 24, 25],
}

for metric in metrics:
    plt.figure(figsize=(16, 12))
    for i, n_cluster in enumerate(n_cluster_list[metric]):
        plt.subplot(5, 5, i + 1)
        plt.title('{} clusters ({})'.format(
            n_cluster, metric_names[metric]))

        x, y = get_xy(15, n_cluster, metric)
        plt.plot(x, y, label='15sec')
        x, y = get_xy(30, n_cluster, metric)
        plt.plot(x, y, label='30sec')

        # plt.xlim(x.min(), x.max())
        plt.ylim(0, 1.1)
        plt.suptitle('{} clusters'.format(n_cluster), y=0)
        plt.xlabel('Distance')
        plt.ylabel('Cumulative accuracy')
        plt.legend()
        plt.tight_layout()

    plt.savefig(
        DATAPATH + '/vis/compare_{}_cluster.png'.format(
            metric, n_cluster))
