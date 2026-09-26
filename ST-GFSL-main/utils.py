import os
import zipfile
import numpy as np
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def mae_np(pred, y):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(y))))


def rmse_np(pred, y):
    diff = np.asarray(pred) - np.asarray(y)
    return float(np.sqrt(np.mean(diff ** 2)))


def mape_np(pred, y, eps=1e-5):
    y = np.asarray(y)
    pred = np.asarray(pred)
    return float(np.mean(np.abs((pred - y) / np.maximum(np.abs(y), eps))) * 100.0)


def masked_mape_np(pred, y, thr=2.0):
    y = np.asarray(y)
    pred = np.asarray(pred)
    mask = np.abs(y) > thr
    if mask.sum() == 0:
        return float('nan'), 0.0
    return float(np.mean(np.abs((pred[mask] - y[mask]) / np.abs(y[mask]))) * 100.0), float(mask.mean())


def metric_func(pred, y, times, mape_mask_threshold=2.0, report_horizons=(3, 6, 12)):
    """
    pred/y: [num_samples, horizon, num_nodes], already on physical scale.

    Overall metrics match STID/OASTID:
      - concat full test set first, then score globally
      - MAE / RMSE over all valid points in [S, H, N]
      - RMSE = sqrt(mean(sq)) over all points (NOT mean of per-step RMSEs)
      - Masked-MAPE drops |y| <= threshold (default 2, same as OASTID mask2)
    """
    pred = np.asarray(pred)
    y = np.asarray(y)
    assert pred.shape == y.shape
    assert pred.shape[1] == times

    result = {
        'MSE': np.zeros(times),
        'RMSE': np.zeros(times),
        'MAE': np.zeros(times),
        'MAPE': np.zeros(times),
    }
    for i in range(times):
        y_i = y[:, i, :]
        pred_i = pred[:, i, :]
        result['MAE'][i] = mae_np(pred_i, y_i)
        result['RMSE'][i] = rmse_np(pred_i, y_i)
        result['MSE'][i] = result['RMSE'][i] ** 2
        result['MAPE'][i] = mape_np(pred_i, y_i) / 100.0  # keep fraction for old printers

    mmape, valid = masked_mape_np(pred, y, mape_mask_threshold)
    result['overall'] = {
        'MAE': mae_np(pred, y),
        'RMSE': rmse_np(pred, y),
        'MAPE%': mape_np(pred, y),
        'Masked-MAPE%': mmape,
        'mape_valid_ratio': valid,
        'mape_mask_threshold': float(mape_mask_threshold),
    }
    # 1-based horizons, e.g. 3/6/12
    result['report_horizons'] = []
    for h in report_horizons:
        if 1 <= h <= times:
            idx = h - 1
            result['report_horizons'].append({
                'h': h,
                'MAE': float(result['MAE'][idx]),
                'RMSE': float(result['RMSE'][idx]),
                'MAPE%': float(result['MAPE'][idx] * 100.0),
            })
    return result


def result_print(result, info_name='Evaluate'):
    total_RMSE, total_MAE, total_MAPE = result['RMSE'], result['MAE'], result['MAPE']
    mae_str = '/ '.join(['%.3f' % v for v in total_MAE])
    mape_str = '/ '.join(['%.3f' % (v * 100) for v in total_MAPE])
    rmse_str = '/ '.join(['%.3f' % v for v in total_RMSE])
    print("========== {} results ==========".format(info_name))
    print(" MAE: " + mae_str)
    print("MAPE: " + mape_str)
    print("RMSE: " + rmse_str)
    if 'overall' in result:
        o = result['overall']
        print("OVERALL(STID/OASTID-style global): MAE=%.4f RMSE=%.4f MAPE=%.4f%% Masked-MAPE@%.0f=%.4f%% (valid=%.4f)" % (
            o['MAE'], o['RMSE'], o['MAPE%'], o['mape_mask_threshold'], o['Masked-MAPE%'], o['mape_valid_ratio']))
        # Equal-weight mean of step MAEs == overall MAE; mean of step RMSEs != overall RMSE
        print(" STEP-MEAN(for reference): MAE=%.4f RMSE=%.4f MAPE=%.4f%%" % (
            float(np.mean(total_MAE)), float(np.mean(total_RMSE)), float(np.mean(total_MAPE) * 100)))
    if result.get('report_horizons'):
        hs = result['report_horizons']
        print(" H={}: MAE= {} | RMSE= {} | MAPE%= {}".format(
            '/'.join(str(x['h']) for x in hs),
            '/ '.join('%.3f' % x['MAE'] for x in hs),
            '/ '.join('%.3f' % x['RMSE'] for x in hs),
            '/ '.join('%.3f' % x['MAPE%'] for x in hs),
        ))
    print("---------------------------------------")


def load_data(dataset_name, stage):
    print("INFO: load {} data @ {} stage".format(dataset_name, stage))

    A = np.load("data/" + dataset_name + "/matrix.npy")
    A = get_normalized_adj(A)
    A = torch.from_numpy(A)
    X = np.load("data/" + dataset_name + "/dataset.npy")
    X = X.transpose((1, 2, 0))
    X = X.astype(np.float32)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    # train: 70%, validation: 10%, test: 20%
    # source: 100%, target_1day: 288, target_3day: 288*3, target_1week: 288*7
    if stage == 'train':
        X = X[:, :, :int(X.shape[2]*0.7)]
    elif stage == 'validation':
        X = X[:, :, int(X.shape[2]*0.7):int(X.shape[2]*0.8)]
    elif stage == 'test':
        X = X[:, :, int(X.shape[2]*0.8):]
    elif stage == 'source':
        X = X
    elif stage == 'target_1day':
        X = X[:, :, :288]
    elif stage == 'target_3day':
        X = X[:, :, :288*3]
    elif stage == 'target_1week':
        X = X[:, :, :288*7]
    else:
        print("Error: unsupported data stage")

    print("INFO: A shape is {}, X shape is {}, means = {}, stds = {}".format(A.shape, X.shape, means, stds))

    return A, X, means, stds


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output, means, stds):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_timesteps_input, num_features).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_timesteps_output).
    """
    total = num_timesteps_input + num_timesteps_output
    num_samples = X.shape[2] - total + 1
    if num_samples <= 0:
        raise ValueError('Not enough timesteps to build samples')

    # Vectorized sliding windows for LargeST-scale sequences.
    windows = np.lib.stride_tricks.sliding_window_view(X, total, axis=2)
    # windows: (N, F, num_samples, total)
    features = np.ascontiguousarray(
        windows[:, :, :, :num_timesteps_input].transpose(2, 0, 3, 1),
        dtype=np.float32)
    target = np.ascontiguousarray(
        windows[:, 0, :, num_timesteps_input:] * stds[0] + means[0],
        dtype=np.float32).transpose(1, 0, 2)

    return torch.from_numpy(features), torch.from_numpy(target)
