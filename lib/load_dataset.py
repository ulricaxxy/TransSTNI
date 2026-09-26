import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def load_st_dataset(dataset):
    if dataset == 'PEMSD4':
        data_path = os.path.join(DATA_DIR, 'pems04.npz')
        data = np.load(data_path)['data'][:, :, 0:1]
    elif dataset == 'PEMSD8':
        data_path = os.path.join(DATA_DIR, 'PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0:1]
    elif dataset == 'PEMSD7':
        data_path = os.path.join(DATA_DIR, 'PEMS07.npz')
        data = np.load(data_path)['data'][:, :, 0:1]
    elif dataset == 'air_quality_full':
        data_path = os.path.join(DATA_DIR, 'air_quality.npz')
        data = np.load(data_path)['data']
    elif dataset == 'pv_us':
        data_path = os.path.join(DATA_DIR, 'pv_us.npz')
        data = np.load(data_path)['data']
    elif dataset == 'korea_covid':
        data_path = os.path.join(DATA_DIR, 'covid_new_t390_n17.npz')
        data = np.load(data_path)['data']
    elif dataset in ('largest_ca', 'largest_gla', 'largest_gba', 'largest_sd'):
        data_path = os.path.join(DATA_DIR, dataset + '.npz')
        data = np.load(data_path)['data'][:, :, 0:1]
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data
