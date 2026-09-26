import torch
import numpy as np
import torch.utils.data
from lib.add_window import Add_Window_Horizon
from lib.load_dataset import load_st_dataset
from lib.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler


def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler


def split_data_by_days(data, val_days, test_days, interval=60):
    T = int((24 * 60) / interval)
    test_data = data[-T * test_days:]
    val_data = data[-T * (test_days + val_days): -T * test_days]
    train_data = data[:-T * (test_days + val_days)]
    return train_data, val_data, test_data


def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len * test_ratio):]
    val_data = data[-int(data_len * (test_ratio + val_ratio)):-int(data_len * test_ratio)]
    train_data = data[:-int(data_len * (test_ratio + val_ratio))]
    return train_data, val_data, test_data


def data_loader(X, Y, batch_size, shuffle=True, drop_last=True, IDX=None):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    if IDX is not None:
        IDX = torch.as_tensor(np.asarray(IDX), dtype=torch.int64)
        if cuda:
            IDX = IDX.cuda()
        data = torch.utils.data.TensorDataset(X, Y, IDX)
    else:
        data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    return dataloader


def get_dataloader(args, normalizer='std', tod=False, dow=False, weather=False, single=True,
                   return_index=False):
    """return_index=True 时，loader 每个 batch 额外返回窗口首步的全局时间下标
    （真实 cycle_index，供 OAMoECycleNet/Hier 做真实日历周期查表）。
    """
    data = load_st_dataset(args.dataset)
    data, scaler = normalize_dataset(data, normalizer, args.column_wise)
    if args.test_ratio > 1:
        data_train, data_val, data_test = split_data_by_days(data, args.val_ratio, args.test_ratio)
    else:
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
    x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
    x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)
    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)

    idx_tra = idx_val = idx_test = None
    if return_index:
        off_val = len(data_train)
        off_test = len(data_train) + len(data_val)
        idx_tra = np.arange(len(x_tra), dtype=np.int64)
        idx_val = np.arange(len(x_val), dtype=np.int64) + off_val
        idx_test = np.arange(len(x_test), dtype=np.int64) + off_test
        print('cycle_index enabled: val offset={}, test offset={}'.format(off_val, off_test))

    train_dataloader = data_loader(
        x_tra, y_tra, args.batch_size, shuffle=True, drop_last=True, IDX=idx_tra)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(
            x_val, y_val, args.batch_size, shuffle=False, drop_last=True, IDX=idx_val)
    test_dataloader = data_loader(
        x_test, y_test, args.batch_size, shuffle=False, drop_last=False, IDX=idx_test)
    return train_dataloader, val_dataloader, test_dataloader, scaler
