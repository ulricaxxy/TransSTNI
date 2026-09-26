import torch
import torch.nn as nn
import numpy as np


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, prediction, target):
        error = torch.abs(prediction - target)
        return error.mean()


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, prediction, target):
        error = (prediction - target) ** 2
        return error.mean()
    

class MAELossMask(nn.Module):
    def forward(self, y_pred, y_true):
        mask = (y_true != 0.0)
        if not mask.any():
            return torch.zeros([], device=y_true.device)
        return torch.abs(y_pred[mask] - y_true[mask]).mean()

def metric(y_pred, y_true):
    mae_3, rmse_3, mape_3 = _metric(y_pred[:, 2, :], y_true[:, 2, :])
    mae_6, rmse_6, mape_6 = _metric(y_pred[:, 5, :], y_true[:, 5, :])
    mae_12, rmse_12, mape_12 = _metric(y_pred[:, 11, :], y_true[:, 11, :])
    mae_avg, rmse_avg, mape_avg = _metric(y_pred, y_true)
    return (mae_3, mae_6, mae_12, mae_avg), (rmse_3, rmse_6, rmse_12, rmse_avg), (mape_3, mape_6, mape_12, mape_avg)

def _metric(y_pred, y_true):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(y_true, 0.0)
        mask = mask.astype(np.float64)
        mask /= np.sum(mask)
        mae = np.abs(np.subtract(y_pred, y_true)).astype(np.float64)
        wape = np.divide(np.sum(mae), np.sum(y_true))
        wape = np.nan_to_num(wape * mask)
        rmse = np.square(mae)
        mape = np.divide(mae, y_true)
        mae = np.nan_to_num(mae * mask)
        mae = np.sum(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.sum(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.sum(mape)
        mape = mape * 100.0
    return mae, rmse, mape


if __name__ == "__main__":
    ...