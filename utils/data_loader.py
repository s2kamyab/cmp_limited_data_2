import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import gaussian_kde
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.fft import fft
# from scipy.signal import cwt, ricker  # Ricker (Mexican hat) is a common wavelet

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

def normalise_selected_columns(window_data,columns_to_normalise, single_window=True):
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            if len(window.shape) == 1:
                window = np.expand_dims(window, axis=0)  # Ensure window is 2D
            for col_i in range(window.shape[1]):
                if col_i in columns_to_normalise:
                    # Normalize only if the column index is in the list of columns to normalize
                    w = window[-1, col_i]
                    if w == 0:
                        w = 1
                    normalised_col = [((float(p) / float(w)) ) for p in window[:, col_i]]
                else:
                    # Keep the original data for columns not in the list
                    normalised_col = window[:, col_i].tolist()
                normalised_window.append(normalised_col)
            normalised_data.append(np.squeeze(np.array(normalised_window).T))
            # normalised_data.append(normalised_window)
        return np.squeeze(np.array(normalised_data))
# def normalise_selected_columns(window_data, columns_to_normalise, single_window=False):
#         # normalised_data = []
#         # window_data = [window_data] if single_window else window_data
#         # for window in window_data:
#         # normalised_window = []
#         # for ts_i in range(window_data.shape[0]):
#             # if col_i in columns_to_normalise:
#                 # Normalize only if the column index is in the list of columns to normalize
#         w = window_data[-2]
#         w[w==0] = 1
#         normalised_window = window_data/w-1#[((p / w) - 1) for p in window_data]
#             # else:
#                 # Keep the original data for columns not in the list
#                 # normalised_col = window[col_i].tolist()
#         # normalised_window.append(normalised_col)
#         # normalised_window = np.array(normalised_window).T
#         # normalised_data.append(normalised_window)
#         return np.array(normalised_window)


def create_seqs_normalized(dfs, common_cols, seq_len, pred_len, normalization, columns_to_normalize, target_index):
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
            # if normalization == 'standard':
            #     means = np.mean(x_seq[:, columns_to_normalize], axis=0)#.mean()
            #     stds = np.std(x_seq[:, columns_to_normalize], axis=0)#.std().replace(0, 1e-8)
            #     stds[stds == 0] = 1e-8  # avoid divide-by-zero
            #     means = np.expand_dims(means, axis=0)
            #     stds = np.expand_dims(stds, axis=0)
            #     # Normalize past values (including target at time t)
            #     x_seq_normalized = x_seq.copy()
            #     x_seq_normalized[:, columns_to_normalize] = (x_seq[:, columns_to_normalize] - np.tile(means, [x_seq[:, columns_to_normalize].shape[0], 1])) / np.tile(stds, [x_seq[:, columns_to_normalize].shape[0], 1])
            #     x_seq_normalized = np.nan_to_num(x_seq_normalized, nan=0)
            #     y_seq_normalized = (y_seq - means[0, target_index]) / stds[0, target_index]
            #     y_seq_normalized = np.nan_to_num(y_seq_normalized, nan=0)
            # elif normalization == 'uniform':
            #     x_seq_normalized = x_seq.copy()
            #     maxs = np.max(x_seq[:, columns_to_normalize], axis =0)
            #     maxs = np.expand_dims(maxs,axis = 0 )
            #     mins = np.min(x_seq[:, columns_to_normalize], axis = 0)
            #     mins = np.expand_dims(mins,axis = 0 )
            #     x_seq_normalized[:, columns_to_normalize] = (x_seq[:, columns_to_normalize] - np.tile(mins, [x_seq.shape[0], 1])) / np.tile(maxs-mins, [x_seq.shape[0], 1])
            #     x_seq_normalized = np.nan_to_num(x_seq_normalized, nan=0)
            #     y_seq_normalized = (y_seq - np.tile(mins[0, target_index], [y_seq.shape[0], 1])) / np.tile(maxs[0, target_index]-mins[0, target_index], [y_seq.shape[0], 1]) #if (maxs[0, target_index]-mins[0, target_index]) != 0 else 1)
            #     y_seq_normalized = np.nan_to_num(y_seq_normalized, nan=0)
            # elif normalization == 'relative':
            #     # x_seq_normalized = x_seq.copy()
            #     t = normalise_selected_columns(values[i:i+seq_len+pred_len], columns_to_normalize, single_window=True)
            #     x_seq_normalized = t[:seq_len, :]
            #     y_seq_normalized = t[seq_len:, target_index]
            # if normalization == 'None':
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

def preprocess(preprocess, train1, test1, target_index):
    if preprocess == 'decompose':
        components = decompose_series(train1.iloc[:,target_index], model='additive', freq=52)

        # Access components
        trend1 = components['trend']
        seasonal1 = components['seasonal']
        residual1 = components['resid']
        train1['residual'] = residual1
        train1['trend'] = trend1
        train1['seasonal'] = seasonal1

        components = decompose_series(test1.iloc[:,target_index], model='additive', freq=52)

        # Access components
        trend1 = components['trend']
        seasonal1 = components['seasonal']
        residual1 = components['resid']
        test1['residual'] = residual1
        test1['trend'] = trend1
        test1['seasonal'] = seasonal1
        target_index = [train1.columns.get_loc('trend'), train1.columns.get_loc('seasonal'), train1.columns.get_loc('residual')]

    elif preprocess == 'fft':
        fft_vals = np.fft.fft(train1.iloc[:,target_index])
        fft_freqs = np.fft.fftfreq(len(fft_vals))
        train1['fft_real'] = np.real(fft_vals)
        train1['fft_mag'] = np.abs(fft_vals)

        fft_vals = np.fft.fft(test1.iloc[:,target_index])
        fft_freqs = np.fft.fftfreq(len(fft_vals))
        test1['fft_real'] = np.real(fft_vals)
        test1['fft_mag'] = np.abs(fft_vals)

        target_index = [train1.columns.get_loc('fft_mag')]
    elif preprocess == 'None':
        target_index = [target_index]
    return train1, test1, target_index
###########################################################################################
# load data
###########################################################################################
def load_data(dataset, preprocess_type, seq_len, pred_len,batch_size, normalization, use_sentiment):
    if dataset == 'soshianest_5627':
        df = pd.read_csv(r'data\\5627_dataset.csv')
        
        df['time_step'] = range(len(df))
        # df = df.drop('date', axis=1)
        columns_to_normalize = range(len(df.columns))
        # columns_to_normalize = [columns_to_normalize[:-2]] # exclude sentiment column
        if use_sentiment == False:
            df = df.drop('Sentiment_textblob', axis=1)
        target_index = df.columns.to_list().index('OT')
        train1, test1 = train_test_split_time_series(df, test_size=0.3)
        

    elif dataset == 'soshianest_530486':
        df = pd.read_csv(r'data\\530486_dataset.csv')
        
        df['time_step'] = range(len(df))
        # df = df.drop('date', axis=1)
        columns_to_normalize = range(len(df.columns))
        # columns_to_normalize = [columns_to_normalize[:-2]] # exclude sentiment column
        if use_sentiment == False:
            df = df.drop('Sentiment_textblob', axis=1)
        target_index = df.columns.to_list().index('OT')  
        train1, test1 = train_test_split_time_series(df, test_size=0.3)

    elif dataset == 'soshianest_530501':
        df = pd.read_csv(r'data\\530501_dataset.csv')
        
        df['time_step'] = range(len(df))
        # df = df.drop('date', axis=1)
        columns_to_normalize = range(len(df.columns))
        # columns_to_normalize = [columns_to_normalize[:-1]] # exclude sentiment column
        if use_sentiment == False:
            df = df.drop('Sentiment_textblob', axis=1)
        target_index = df.columns.to_list().index('OT')    
        train1, test1 = train_test_split_time_series(df, test_size=0.3)

    elif dataset == 'soshianest_549324':
        df = pd.read_csv(r'data\\549324_dataset.csv')
        
        df['time_step'] = range(len(df))
        # df = df.drop('date', axis=1)
        columns_to_normalize = range(len(df.columns))
        # columns_to_normalize = [columns_to_normalize[:-1]] # exclude sentiment column
        if use_sentiment == False:
            df = df.drop('Sentiment_textblob', axis=1)
        target_index = df.columns.to_list().index('OT')    
        train1, test1 = train_test_split_time_series(df, test_size=0.3)

    elif dataset == 'fin_aal':
        df = pd.read_csv(r'data\\aal.csv')
        
        df['time_step'] = range(len(df))
        df = df.drop('Date', axis=1)
        df = df.drop('News_flag', axis=1)
        
        if use_sentiment == False:
            df = df.drop('Scaled_sentiment', axis=1)
            df = df.drop('Sentiment_gpt', axis=1)
        # Columns to exclude
        exclude_cols = []#['Scaled_sentiment', 'Sentiment_gpt']
        # Get indexes of columns to keep
        include_cols = [col for col in df.columns if col not in exclude_cols]
        columns_to_normalize = [df.columns.get_loc(col) for col in include_cols]  
        target_index = df.columns.to_list().index('Close')
        train1, test1 = train_test_split_time_series(df, test_size=0.3)


    elif dataset == 'fin_aapl':
        df = pd.read_csv(r'data\\AAPL.csv')
        # df = df.loc[:,['Date', 'Close', 'Volume', 'Scaled_sentiment']]
        df['time_step'] = range(len(df))
        df = df.drop('Date', axis=1)
        df = df.drop('News_flag', axis=1)
        
        if use_sentiment == False:
            df = df.drop('Scaled_sentiment', axis=1)
            df = df.drop('Sentiment_gpt', axis=1)
        # Columns to exclude
        exclude_cols =[]# ['Scaled_sentiment', 'Sentiment_gpt']
        # Get indexes of columns to keep
        include_cols = [col for col in df.columns if col not in exclude_cols]
        cols = df.columns#[df.columns.get_loc(col) for col in include_cols]  
        columns_to_normalize = [df.columns.get_loc(col) for col in cols]  
        target_index = df.columns.to_list().index('Close')
        train1, test1 = train_test_split_time_series(df, test_size=0.3)

    elif dataset == 'fin_abbv':
        df = pd.read_csv(r'data\\ABBV.csv')
        
        df['time_step'] = range(len(df))
        df = df.drop('Date', axis=1)
        df = df.drop('News_flag', axis=1)
        
        if use_sentiment == False:
            df = df.drop('Scaled_sentiment', axis=1)
            df = df.drop('Sentiment_gpt', axis=1)
        # Columns to exclude
        exclude_cols = []#['Scaled_sentiment', 'Sentiment_gpt']
        # Get indexes of columns to keep
        include_cols = [col for col in df.columns if col not in exclude_cols]
        columns_to_normalize = [df.columns.get_loc(col) for col in include_cols] 
        target_index = df.columns.to_list().index('Close')
        train1, test1 = train_test_split_time_series(df, test_size=0.3)

    elif dataset == 'fin_amd':
        df = pd.read_csv(r'data\\AMD.csv')
        target_index = df.columns.to_list().index('Close')
        df['time_step'] = range(len(df))
        df = df.drop('Date', axis=1)
        df = df.drop('News_flag', axis=1)

        if use_sentiment == False:
            df = df.drop('Scaled_sentiment', axis=1)
            df = df.drop('Sentiment_gpt', axis=1)
        # Columns to exclude
        exclude_cols = []#['Scaled_sentiment', 'Sentiment_gpt']
        # Get indexes of columns to keep
        include_cols = [col for col in df.columns if col not in exclude_cols]
        columns_to_normalize = [df.columns.get_loc(col) for col in include_cols] 
        target_index = df.columns.to_list().index('Close')
        train1, test1 = train_test_split_time_series(df, test_size=0.3)

    elif dataset == 'fin_ko':
        df = pd.read_csv(r'data\\KO.csv')
        df['time_step'] = range(len(df))
        df = df.drop('Date', axis=1)
        df = df.drop('News_flag', axis=1)
        
        if use_sentiment == False:
            df = df.drop('Scaled_sentiment', axis=1)
            df = df.drop('Sentiment_gpt', axis=1)

        exclude_cols = []#['Scaled_sentiment', 'Sentiment_gpt']
        # Get indexes of columns to keep
        include_cols = [col for col in df.columns if col not in exclude_cols]
        columns_to_normalize = [df.columns.get_loc(col) for col in include_cols] 
        target_index = df.columns.to_list().index('Close')
        train1, test1 = train_test_split_time_series(df, test_size=0.3)

    elif dataset == 'fin_TSM':
        df = pd.read_csv(r'data\\TSM.csv')
        target_index = df.columns.to_list().index('Close')
        df['time_step'] = range(len(df))
        df = df.drop('Date', axis=1)
        df = df.drop('News_flag', axis=1)
        if use_sentiment == False:
            df = df.drop('Scaled_sentiment', axis=1)
            df = df.drop('Sentiment_gpt', axis=1)

        exclude_cols = []#['Scaled_sentiment', 'Sentiment_gpt']
        # Get indexes of columns to keep
        include_cols = [col for col in df.columns if col not in exclude_cols]
        columns_to_normalize = [df.columns.get_loc(col) for col in include_cols] 
        target_index = df.columns.to_list().index('Close')
        train1, test1 = train_test_split_time_series(df, test_size=0.3)

    elif dataset == 'goog':
        df = pd.read_csv(r'data\\GOOG.csv')
        target_index = df.columns.to_list().index('Close')
        df['time_step'] = range(len(df))
        df = df.drop('Date', axis=1)
        df = df.drop('News_flag', axis=1)
        if use_sentiment == False:
            df = df.drop('Scaled_sentiment', axis=1)
            df = df.drop('Sentiment_gpt', axis=1)

        exclude_cols = []#['Scaled_sentiment', 'Sentiment_gpt']
        # Get indexes of columns to keep
        include_cols = [col for col in df.columns if col not in exclude_cols]
        columns_to_normalize = [df.columns.get_loc(col) for col in include_cols] 
        target_index = df.columns.to_list().index('Close')
        train1, test1 = train_test_split_time_series(df, test_size=0.3)

    elif dataset == 'fin_wmt':
        df = pd.read_csv(r'data\\WMT.csv')
        target_index = df.columns.to_list().index('Close')
        df['time_step'] = range(len(df))
        df = df.drop('Date', axis=1)
        df = df.drop('News_flag', axis=1)
        if use_sentiment == False:
            df = df.drop('Scaled_sentiment', axis=1)
            df = df.drop('Sentiment_gpt', axis=1)

        exclude_cols = []#['Scaled_sentiment', 'Sentiment_gpt']
        # Get indexes of columns to keep
        include_cols = [col for col in df.columns if col not in exclude_cols]
        columns_to_normalize = [df.columns.get_loc(col) for col in include_cols] 
        target_index = df.columns.to_list().index('Close')
        train1, test1 = train_test_split_time_series(df, test_size=0.3)
    if preprocess_type == 'decompose':
        output_dim = 3  # trend, seasonal, residual
    else:
        output_dim = 1
    train1, test1, target_index = preprocess(preprocess_type, train1, test1, target_index)   
    if preprocess_type == 'decompose':
        columns_to_normalize = range(len(train1.columns))#[train1.columns.get_loc('trend'), train1.columns.get_loc('seasonal'), train1.columns.get_loc('residual')]
    train_dataset, input_dim, train_dataset_actual = create_seqs_normalized([train1],train1.columns, seq_len, pred_len, normalization, columns_to_normalize, target_index)
    test_dataset, input_dim, test_dataset_actual = create_seqs_normalized([test1],test1.columns, seq_len, pred_len, normalization, columns_to_normalize, target_index)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader_actual = DataLoader(train_dataset_actual, batch_size=batch_size, shuffle=True)
    test_loader_actual = DataLoader(test_dataset_actual, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # if eda:
    #     plot_train_test_target_distributions(train_loader, test_loader, num_outputs=len(target_index))
    return train_loader, test_loader, train_loader_actual, test_loader_actual, input_dim, output_dim, train1.columns.tolist(), target_index, columns_to_normalize
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
