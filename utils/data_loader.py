import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import gaussian_kde
from statsmodels.tsa.seasonal import seasonal_decompose
from utils.EDA import plot_train_test_target_distributions
def decompose_series(series, model='additive', freq=None):
    """
    Decomposes a time series into trend, seasonal, and residual components.
    
    Args:
        series: Pandas Series with datetime index.
        model: 'additive' or 'multiplicative'
        freq: seasonal period (e.g., 52 for weekly with yearly seasonality)
    
    Returns:
        A dict with trend, seasonal, and residual components.
    """
    result = seasonal_decompose(series, model=model, period=freq, extrapolate_trend='freq')
    return {
        'observed': result.observed,
        'trend': result.trend,
        'seasonal': result.seasonal,
        'resid': result.resid
    }
def create_seqs_normalized(dfs, common_cols, seq_len, pred_len, normalization, target_index):
    datasets = []
    datasets_actual = []
    # target_index = [common_cols.index('trend'), common_cols.index('seasonal'), common_cols.index('residual')]

    for df in dfs:
        values = df.values
        x_list, y_list = [], []
        x_list_actual, y_list_actual = [], []
        for i in range(len(values) - seq_len - pred_len):
            x_seq = values[i:i+seq_len, :]
            y_seq = values[i+seq_len: i+seq_len+pred_len, target_index]#values[i+1:i+seq_len+1, target_index]  # just target feature
            # Calculate causal stats up to t (inclusive)
            if normalization == 'standard':
                means = np.mean(x_seq, axis=0)#.mean()
                stds = np.std(x_seq, axis=0)#.std().replace(0, 1e-8)
                stds[stds == 0] = 1e-8  # avoid divide-by-zero
                means = np.expand_dims(means, axis=0)
                stds = np.expand_dims(stds, axis=0)
                # Normalize past values (including target at time t)
                x_seq_normalized = (x_seq - np.tile(means, [x_seq.shape[0], 1])) / np.tile(stds, [x_seq.shape[0], 1])
                x_seq_normalized = np.nan_to_num(x_seq_normalized, nan=0)
                y_seq_normalized = (y_seq - means[0, target_index]) / stds[0, target_index]
            elif normalization == 'uniform':
                maxs = np.max(x_seq, axis =0)
                maxs = np.expand_dims(maxs,axis = 0 )
                mins = np.min(x_seq, axis = 0)
                mins = np.expand_dims(mins,axis = 0 )
                x_seq_normalized = (x_seq - np.tile(mins, [x_seq.shape[0], 1])) / np.tile(maxs-mins, [x_seq.shape[0], 1])
                x_seq_normalized = np.nan_to_num(x_seq_normalized, nan=0)
                y_seq_normalized = (y_seq - mins[0, target_index]) /  np.tile(maxs-mins, [y_seq.shape[0], 1])
            elif normalization == 'None': 
                x_seq_normalized = x_seq
                y_seq_normalized = y_seq

                
            x_list.append(x_seq_normalized)
            y_list.append(y_seq_normalized)
            x_list_actual.append(x_seq)
            y_list_actual.append(y_seq)

        x_tensor = torch.tensor(np.array(x_list), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32)
        datasets.append(torch.utils.data.TensorDataset(x_tensor, y_tensor))
        x_tensor_actual = torch.tensor(np.array(x_list_actual), dtype=torch.float32)
        y_tensor_actual = torch.tensor(np.array(y_list_actual), dtype=torch.float32)
        datasets_actual.append(torch.utils.data.TensorDataset(x_tensor_actual, y_tensor_actual))

    combined = torch.utils.data.ConcatDataset(datasets)
    combined_actual = torch.utils.data.ConcatDataset(datasets_actual)
    return combined, len(common_cols), combined_actual

def train_test_split_time_series(df, test_size=0.1):
    """
    Splits a time series dataframe into train and test sets by time order.
    """
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df

def preprocess(preprocess_type, train1, test1, target_index):
    if preprocess == 'decompose':
        target_index = [1,2,3]
    elif preprocess == 'None':
        target_index = [target_index]
    return train1, test1, target_index

def load_data(dataset, preprocess_type, seq_len, pred_len,batch_size, normalization, eda):
    if dataset == 'soshianest_5627':
        df = pd.read_csv(r'data\\5627_dataset.csv')
        target_index = df.index('OT')
        train1, test1 = train_test_split_time_series(df, test_size=0.1)

    elif dataset == 'soshianest_530486':
        df = pd.read_csv(r'data\\530486_dataset.csv')
        target_index = df.index('OT')
        train1, test1 = train_test_split_time_series(df, test_size=0.1)

    elif dataset == 'soshianest_530501':
        df = pd.read_csv(r'data\\530501_dataset.csv')
        target_index = df.index('OT')
        train1, test1 = train_test_split_time_series(df, test_size=0.1)

    elif dataset == 'soshianest_549324':
        df = pd.read_csv(r'data\\549324_dataset.csv')
        target_index = df.index('OT')
        train1, test1 = train_test_split_time_series(df, test_size=0.1)

    elif dataset == 'fin_aal':
        df = pd.read_csv(r'data\\aal.csv')
        target_index = df.index('Close')
        train1, test1 = train_test_split_time_series(df, test_size=0.1)

    elif dataset == 'fin_aapl':
        df = pd.read_csv(r'data\\AAPL.csv')
        target_index = df.index('Close')
        train1, test1 = train_test_split_time_series(df, test_size=0.1)

    elif dataset == 'fin_abbv':
        target_index = df.index('Close')
        df = pd.read_csv(r'data\\AABV.csv')
        train1, test1 = train_test_split_time_series(df, test_size=0.1)

    elif dataset == 'fin_amd':
        target_index = df.index('Close')
        df = pd.read_csv(r'data\\AMD.csv')
        train1, test1 = train_test_split_time_series(df, test_size=0.1)

    elif dataset == 'fin_ko':
        target_index = df.index('Close')
        df = pd.read_csv(r'data\\KO.csv')
        train1, test1 = train_test_split_time_series(df, test_size=0.1)

    elif dataset == 'fin_TSM':
        target_index = df.index('Close')
        df = pd.read_csv(r'data\\TSM.csv')
        train1, test1 = train_test_split_time_series(df, test_size=0.1)

    elif dataset == 'goog':
        target_index = df.index('Close')
        df = pd.read_csv(r'data\\GOOG.csv')
        train1, test1 = train_test_split_time_series(df, test_size=0.1)

    elif dataset == 'fin_wmt':
        target_index = df.index('Close')
        df = pd.read_csv(r'data\\WMT.csv')
        train1, test1 = train_test_split_time_series(df, test_size=0.1)

    train1, test1, target_index = preprocess(preprocess_type, train1, test1, target_index)   
    train_dataset, input_dim, train_dataset_actual = create_seqs_normalized([train1],train1.columns, seq_len, pred_len, normalization, target_index)
    test_dataset, input_dim, test_dataset_actual = create_seqs_normalized([test1],test1.columns, seq_len, pred_len, normalization, target_index)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader_actual = DataLoader(train_dataset_actual, batch_size=batch_size, shuffle=True)
    test_loader_actual = DataLoader(test_dataset_actual, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    if eda:
        plot_train_test_target_distributions(train_loader, test_loader, num_outputs=len(target_index))
    return train_loader, test_loader, train_loader_actual, test_loader_actual, input_dim
# # Test csvs = 50
#     names_50 = ['aal.csv', 'AAPL.csv', 'ABBV.csv', 'AMD.csv', 'amgn.csv', 'AMZN.csv', 'BABA.csv',
#                 'bhp.csv', 'bidu.csv', 'biib.csv', 'BRK-B.csv', 'C.csv', 'cat.csv', 'cmcsa.csv', 'cmg.csv',
#                 'cop.csv', 'COST.csv', 'crm.csv', 'CVX.csv', 'dal.csv', 'DIS.csv', 'ebay.csv', 'GE.csv',
#                 'gild.csv', 'gld.csv', 'GOOG.csv', 'gsk.csv', 'INTC.csv', 'KO.csv', 'mrk.csv', 'MSFT.csv',
#                 'mu.csv', 'nke.csv', 'nvda.csv', 'orcl.csv', 'pep.csv', 'pypl.csv', 'qcom.csv', 'QQQ.csv',
#                 'SBUX.csv', 'T.csv', 'tgt.csv', 'tm.csv', 'TSLA.csv', 'TSM.csv', 'uso.csv', 'v.csv', 'WFC.csv',
#                 'WMT.csv', 'xlf.csv']

#     # Test csvs = 25
#     names_25 = ['AAPL.csv', 'ABBV.csv', 'AMZN.csv', 'BABA.csv', 'BRK-B.csv', 'C.csv', 'COST.csv', 'CVX.csv', 'DIS.csv',
#                 'GE.csv',
#                 'INTC.csv', 'MSFT.csv', 'nvda.csv', 'pypl.csv', 'QQQ.csv', 'SBUX.csv', 'T.csv', 'TSLA.csv', 'WFC.csv',
#                 'KO.csv', 'AMD.csv', 'TSM.csv', 'GOOG.csv', 'WMT.csv']

#     # Test csvs = 5
#     names_5 = ['KO.csv', 'AMD.csv', 'TSM.csv', 'GOOG.csv', 'WMT.csv']
