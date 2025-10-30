import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import gaussian_kde
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.fft import fft
from tsaug import AddNoise, Crop, Drift, TimeWarp
from scipy.interpolate import interp1d
from tslearn.barycenters import dtw_barycenter_averaging

def dba_augment_single(sequence, std, n_variants=5, seed=None):
    """
    Perform DBA-based augmentation starting from a single sequence.

    Parameters:
        sequence (np.ndarray): Array of shape (T, C)
        n_variants (int): Number of noisy variants to generate
        sigma (float): Jittering noise scale
        seed (int): Optional random seed

    Returns:
        np.ndarray: A single synthetic sequence of shape (T, C)
    """
    sigma = 0.1*std
    if seed is not None:
        np.random.seed(seed)

    T, C = sequence.shape
    variants = np.stack([jitter(sequence, sigma) for _ in range(n_variants)], axis=0)  # shape: (n_variants, T, C)

    synthetic = dtw_barycenter_averaging(variants)
    return synthetic  # shape: (T, C)

def moving_block_bootstrap(x, block_size=5, seed=None):
    """
    Apply Moving Block Bootstrap (MBB) to a multivariate time series.

    Parameters:
        x (np.ndarray): Time series of shape (T, C)
        block_size (int): Size of each block to sample
        seed (int or None): Random seed for reproducibility

    Returns:
        np.ndarray: Resampled time series of shape (T, C)
    """
    if seed is not None:
        np.random.seed(seed)

    T, C = x.shape
    n_blocks = int(np.ceil(T / block_size))
    
    blocks = []
    for _ in range(n_blocks):
        start = np.random.randint(0, T - block_size + 1)
        block = x[start:start + block_size]
        blocks.append(block)

    x_boot = np.concatenate(blocks, axis=0)[:T]  # Truncate to original length
    return x_boot

def jitter(x, std):
    sigma = 0.1*std
    noise = np.random.normal(loc=0.0, scale=sigma, size=x.shape)
    return x + noise
def time_slicing(x, crop_size):
    T = x.shape[0]
    if crop_size >= T:
        return x
    start = np.random.randint(0, T - crop_size)
    return x[start:start + crop_size]
def magnitude_warp(x, std, knot=4):
    x_mg = np.exp(0.01*np.random.rand()) * x
    return x_mg


def rotation(x, k=None):
    T = x.shape[0]
    if k is None:
        k = np.random.randint(1, T)
    return np.roll(x, shift=k, axis=0)


def augment_ts(X, Y, w_augment):#w_jittering=1.0, w_crop=1.0, w_mag_warp=1.0, w_time_warp=1.0, w_rotation=1.0, w_rand_perm=1.0, w_mbb = 1.0):
    """
    Apply selected augmentations to a batch of sequences.

    Args:
        X: np.ndarray of shape (N, T, C)
        w_*: weights (0 or 1) to toggle each augmentation

    Returns:
        np.ndarray of augmented sequences with shape (N, T, C)
    """
    w_jittering = w_augment['w_jit']
    w_crop = w_augment['w_crop']
    w_mag_warp = w_augment['w_mag_warp']
    w_time_warp = w_augment['w_time_warp']
    w_rand_perm=w_augment['w_rand_perm']
    w_mbb = w_augment['w_mbb']
    w_dbn = w_augment['w_dbn']
    w_rotation = w_augment['w_rotation']
    N = X.shape[0]
    augmented_X = []
    augmented_y = []
    
    for i in range(N):

        x = X[i]  # shape (T, C)
        std_x = np.std(x, axis = 0)
        augmented_X.append(x)
        augmented_y.append(Y[i])
        if np.random.rand() < w_jittering:
            x_jittered = jitter(x, std_x)
            augmented_X.append(x_jittered)
            augmented_y.append(Y[i])
        # if np.random.rand() < w_crop:
        #     x_cropped = time_slicing(x, crop_size=len(x))
        #     augmented_X.append(x_cropped)
        if np.random.rand() < w_mag_warp:
            x_mw = magnitude_warp(x, std_x)
            augmented_X.append(x_mw)
            augmented_y.append(Y[i])
        if np.random.rand() < w_mbb:
            x_mbb = moving_block_bootstrap(x, block_size=5, seed=None)
            augmented_X.append(x_mbb)
            augmented_y.append(Y[i])
        if np.random.rand() < w_dbn:
            x_dbn = dba_augment_single(x,std_x, n_variants=5, seed=None)
            augmented_X.append(x_dbn)
            augmented_y.append(Y[i])

        # if np.random.rand() < w_time_warp:
        #     x_tw = time_warp(x)
        #     augmented_X.append(x_tw)
        if np.random.rand() < w_rotation:
            x_rot = rotation(x)
            augmented_X.append(x_rot)
            augmented_y.append(Y[i])
    return np.array(augmented_X), np.array(augmented_y)
    



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

def create_seqs_normalized(dfs, common_cols, 
                           seq_len, pred_len, normalization, columns_to_normalize, 
                           target_index, w_augment):
    datasets = []
    datasets_actual = []
    # target_index = [common_cols.index('trend'), common_cols.index('seasonal'), common_cols.index('residual')]

    for df in dfs:
        values = df.values
                
        x_list, y_list = [], []
        x_list_actual, y_list_actual = [], []
        for i in range(len(values) - seq_len - pred_len):
            x_seq = values[i:i+seq_len, target_index]
            y_seq = values[i+seq_len: i+seq_len+pred_len, target_index]#values[i+1:i+seq_len+1, target_index]  # just target feature
            x_seq_normalized = x_seq
            y_seq_normalized = y_seq

            x_list.append(x_seq_normalized)
            y_list.append(y_seq_normalized)
            x_list_actual.append(x_seq)
            y_list_actual.append(y_seq)
        x_array = np.array(x_list)
        y_array = np.array(y_list)
        x_array, y_array = augment_ts(x_array, y_array, w_augment) 

        x_tensor = torch.tensor(x_array, dtype=torch.float32)
        y_tensor = torch.tensor(y_array, dtype=torch.float32)

        # x_tensor = torch.tensor(np.array(x_list), dtype=torch.float32)
        # y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32)
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
        # test1['residual'] = residual1
        test1['trend'] = trend1
        test1['seasonal'] = seasonal1
        target_index = [train1.columns.get_loc('trend'), train1.columns.get_loc('seasonal')]#, train1.columns.get_loc('residual')]

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

def handle_outliers(df, target_col, method="zscore", threshold=3.0, replace_with="mean"):
    """
    Detect and replace outliers in a target column while keeping them separately.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str
        Column name containing the target values.
    method : str
        Outlier detection method: "zscore" or "iqr".
    threshold : float
        Z-score threshold (if method="zscore") or IQR multiplier (if method="iqr").
    replace_with : str
        Replacement value for outliers: "mean" or "median".

    Returns
    -------
    df_clean : pd.DataFrame
        DataFrame with outliers replaced.
    df_outliers : pd.DataFrame
        DataFrame containing only the outlier rows.
    """

    x = df[target_col].astype(float)

    if method == "zscore":
        zscores = (x - x.mean()) / x.std(ddof=0)
        mask = np.abs(zscores) > threshold
    elif method == "iqr":
        Q1, Q3 = x.quantile(0.25), x.quantile(0.75)
        IQR = Q3 - Q1
        mask = (x < Q1 - threshold * IQR) | (x > Q3 + threshold * IQR)
    else:
        raise ValueError("method must be 'zscore' or 'iqr'")

    # Save outliers separately
    df_outliers = df[mask].copy()

    # Replace outliers with average
    replacement_value = x.mean() if replace_with == "mean" else x.median()
    df_clean = df.copy()
    df_clean.loc[mask, target_col] = replacement_value

    return df_clean, df_outliers
###########################################################################################
# load data
###########################################################################################
def load_data_pretrain(dataset, 
              preprocess_type,
                seq_len, 
                pred_len,
                batch_size,
                  normalization,
                    use_sentiment,
                      w_aug):
    if dataset == 'soshianest_5627' or dataset == '5627_dataset.csv':
        df = pd.read_csv(r'paper_1_git_repo/data_soshianest/5627_dataset_with_sentiment.csv')
        df = df.drop('txtdate', axis=1)
        # df = df.drop('14780', axis=1)
        df['time_step'] = range(len(df))
        # df = df.drop('date', axis=1)
        
        # columns_to_normalize = [columns_to_normalize[:-2]] # exclude sentiment column
        if use_sentiment == 0:
            df = df.drop('Sentiment_textblob', axis=1)
        else:
            df['Sentiment_textblob'] = df['Sentiment_textblob'].shift(use_sentiment).fillna(0)

        # remove constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        for cols in constant_cols:
            df.drop(columns=cols, inplace=True)
        target_name = 'OT'
        target_index = df.columns.to_list().index(target_name)
        columns_to_normalize = range(len(df.columns))
        components = decompose_series(df.iloc[:,target_index], model='additive', freq=52)

        # Access components
        trend1 = components['trend']
        seasonal1 = components['seasonal']
        residual1 = components['resid']
        df[target_name] = trend1 + seasonal1 # predict trend instead of the whole
        train1, test1 = train_test_split_time_series(df, test_size=0.3)
        
    elif dataset == 'soshianest_530486'or dataset == '530486_dataset.csv':
        df = pd.read_csv(r'paper_1_git_repo/data_soshianest/530486_dataset_with_sentiment.csv')
        df = df.drop('txtdate', axis=1)
        df = df.drop('14780', axis=1)
        df['time_step'] = range(len(df))
        # df = df.drop('date', axis=1)
        
        # columns_to_normalize = [columns_to_normalize[:-2]] # exclude sentiment column
        if use_sentiment == 0:
            df = df.drop('Sentiment_textblob', axis=1)
        else:
            df['Sentiment_textblob'] = df['Sentiment_textblob'].shift(use_sentiment).fillna(0)
        # remove constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        for cols in constant_cols:
            df.drop(columns=cols, inplace=True)
        target_name = 'OT'
        target_index = df.columns.to_list().index(target_name)
        columns_to_normalize = range(len(df.columns))
        components = decompose_series(df.iloc[:,target_index], model='additive', freq=52)

        # Access components
        trend1 = components['trend']
        seasonal1 = components['seasonal']
        residual1 = components['resid']
        df[target_name] = trend1 + seasonal1 # predict trend instead of the whole

        train1, test1 = train_test_split_time_series(df, test_size=0.3)

    elif dataset == 'soshianest_530501' or dataset == '530501_dataset.csv':
        df = pd.read_csv(r'paper_1_git_repo/data_soshianest/530501_dataset_with_sentiment.csv')
        # df = df[['OT', 'Sentiment_textblob']]
        df = df.drop('txtdate', axis=1)
        # df = df.drop('14780', axis=1)
        df['time_step'] = range(len(df))
        # df = df.drop('date', axis=1)
        
        # columns_to_normalize = [columns_to_normalize[:-1]] # exclude sentiment column
        if use_sentiment == 0:
            df = df.drop('Sentiment_textblob', axis=1)
        else:
            df['Sentiment_textblob'] = df['Sentiment_textblob'].shift(use_sentiment).fillna(0)
        
        # remove constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        for cols in constant_cols:
            df.drop(columns=cols, inplace=True)
        target_name = 'OT'
        target_index = df.columns.to_list().index(target_name)
        columns_to_normalize = range(len(df.columns))
        components = decompose_series(df.iloc[:,target_index], model='additive', freq=52)

        # Access components
        trend1 = components['trend']
        seasonal1 = components['seasonal']
        residual1 = components['resid']
        df[target_name] = trend1 + seasonal1 # predict trend instead of the whole

        train1, test1 = train_test_split_time_series(df, test_size=0.3)

    elif dataset == 'soshianest_549324' or dataset == '549324_dataset.csv' :
        df = pd.read_csv(r'paper_1_git_repo/data_soshianest/549324_dataset_with_sentiment.csv')
        df = df.drop('txtdate', axis=1)
        # df = df.drop('14780', axis=1)
        df['time_step'] = range(len(df))
        # df = df.drop('date', axis=1)
        
        # columns_to_normalize = [columns_to_normalize[:-1]] # exclude sentiment column
        if use_sentiment == 0:
            df = df.drop('Sentiment_textblob', axis=1)
        else:
            df['Sentiment_textblob'] = df['Sentiment_textblob'].shift(use_sentiment).fillna(0)
        
        # remove constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        for cols in constant_cols:
            df.drop(columns=cols, inplace=True)
        target_name = 'OT'
        target_index = df.columns.to_list().index(target_name)
        columns_to_normalize = range(len(df.columns))
        components = decompose_series(df.iloc[:,target_index], model='additive', freq=52)

        # Access components
        trend1 = components['trend']
        seasonal1 = components['seasonal']
        residual1 = components['resid']
        df[target_name] = trend1 + seasonal1 # predict trend instead of the whole

        train1, test1 = train_test_split_time_series(df, test_size=0.3)

    elif dataset == 'fin_aal' or dataset == 'aal.csv':
        df = pd.read_csv(r'data\\aal.csv')
        
        df['time_step'] = range(len(df))
        df = df.drop('Date', axis=1)
        df = df.drop('News_flag', axis=1)
        
        if use_sentiment == 0:
            df = df.drop('Scaled_sentiment', axis=1)
            df = df.drop('Sentiment_gpt', axis=1)
        else:
            df['Scaled_sentiment'] = df['Scaled_sentiment'].shift(use_sentiment).fillna(0)
            df['Sentiment_gpt'] = df['Sentiment_gpt'].shift(use_sentiment).fillna(0)
                
        # Columns to exclude
        exclude_cols = []#['Scaled_sentiment', 'Sentiment_gpt']
        # Get indexes of columns to keep
        include_cols = [col for col in df.columns if col not in exclude_cols]
        columns_to_normalize = [df.columns.get_loc(col) for col in include_cols]  
        target_index = df.columns.to_list().index('Close')
        train1, test1 = train_test_split_time_series(df, test_size=0.3)


    elif dataset == 'fin_aapl' or dataset == 'AAPL.csv':
        df = pd.read_csv(r'data\\AAPL.csv')
        # df = df.loc[:,['Date', 'Close', 'Volume', 'Scaled_sentiment']]
        df['time_step'] = range(len(df))
        df = df.drop('Date', axis=1)
        df = df.drop('News_flag', axis=1)
        
        if use_sentiment == 0:
            df = df.drop('Scaled_sentiment', axis=1)
            df = df.drop('Sentiment_gpt', axis=1)
        else:
            df['Scaled_sentiment'] = df['Scaled_sentiment'].shift(use_sentiment).fillna(0)
            df['Sentiment_gpt'] = df['Sentiment_gpt'].shift(use_sentiment).fillna(0)
        # Columns to exclude
        exclude_cols =[]# ['Scaled_sentiment', 'Sentiment_gpt']
        # Get indexes of columns to keep
        include_cols = [col for col in df.columns if col not in exclude_cols]
        cols = df.columns#[df.columns.get_loc(col) for col in include_cols]  
        columns_to_normalize = [df.columns.get_loc(col) for col in cols]  
        target_index = df.columns.to_list().index('Close')
        train1, test1 = train_test_split_time_series(df, test_size=0.3)

    elif dataset == 'fin_abbv' or dataset == 'ABBV.csv':
        df = pd.read_csv(r'data\\ABBV.csv')
        
        df['time_step'] = range(len(df))
        df = df.drop('Date', axis=1)
        df = df.drop('News_flag', axis=1)
        
        if use_sentiment == 0:
            df = df.drop('Scaled_sentiment', axis=1)
            df = df.drop('Sentiment_gpt', axis=1)
        else:
            df['Scaled_sentiment'] = df['Scaled_sentiment'].shift(use_sentiment).fillna(0)
            df['Sentiment_gpt'] = df['Sentiment_gpt'].shift(use_sentiment).fillna(0)
        # Columns to exclude
        exclude_cols = []#['Scaled_sentiment', 'Sentiment_gpt']
        # Get indexes of columns to keep
        include_cols = [col for col in df.columns if col not in exclude_cols]
        columns_to_normalize = [df.columns.get_loc(col) for col in include_cols] 
        target_index = df.columns.to_list().index('Close')
        train1, test1 = train_test_split_time_series(df, test_size=0.3)

    elif dataset == 'fin_amd' or dataset == 'AMD.csv':
        df = pd.read_csv(r'data\\AMD.csv')
        target_index = df.columns.to_list().index('Close')
        df['time_step'] = range(len(df))
        df = df.drop('Date', axis=1)
        df = df.drop('News_flag', axis=1)

        if use_sentiment == 0:
            df = df.drop('Scaled_sentiment', axis=1)
            df = df.drop('Sentiment_gpt', axis=1)
        else:
            df['Scaled_sentiment'] = df['Scaled_sentiment'].shift(use_sentiment).fillna(0)
            df['Sentiment_gpt'] = df['Sentiment_gpt'].shift(use_sentiment).fillna(0)
        # Columns to exclude
        exclude_cols = []#['Scaled_sentiment', 'Sentiment_gpt']
        # Get indexes of columns to keep
        include_cols = [col for col in df.columns if col not in exclude_cols]
        columns_to_normalize = [df.columns.get_loc(col) for col in include_cols] 
        target_index = df.columns.to_list().index('Close')
        train1, test1 = train_test_split_time_series(df, test_size=0.3)

    elif dataset == 'fin_ko' or dataset == 'KO.csv':
        df = pd.read_csv(r'data\\KO.csv')
        df['time_step'] = range(len(df))
        df = df.drop('Date', axis=1)
        df = df.drop('News_flag', axis=1)
        
        if use_sentiment == 0:
            df = df.drop('Scaled_sentiment', axis=1)
            df = df.drop('Sentiment_gpt', axis=1)
        else:
            df['Scaled_sentiment'] = df['Scaled_sentiment'].shift(use_sentiment).fillna(0)
            df['Sentiment_gpt'] = df['Sentiment_gpt'].shift(use_sentiment).fillna(0)
        exclude_cols = []#['Scaled_sentiment', 'Sentiment_gpt']
        # Get indexes of columns to keep
        include_cols = [col for col in df.columns if col not in exclude_cols]
        columns_to_normalize = [df.columns.get_loc(col) for col in include_cols] 
        target_index = df.columns.to_list().index('Close')
        train1, test1 = train_test_split_time_series(df, test_size=0.3)

    elif dataset == 'fin_TSM'or dataset == 'TSM.csv':
        df = pd.read_csv(r'data\\TSM.csv')
        df['time_step'] = range(len(df))
        df = df.drop('Date', axis=1)
        df = df.drop('News_flag', axis=1)
        if use_sentiment == 0:
            df = df.drop('Scaled_sentiment', axis=1)
            df = df.drop('Sentiment_gpt', axis=1)
        else:
            df['Scaled_sentiment'] = df['Scaled_sentiment'].shift(use_sentiment).fillna(0)
            df['Sentiment_gpt'] = df['Sentiment_gpt'].shift(use_sentiment).fillna(0)

        exclude_cols = []#['Scaled_sentiment', 'Sentiment_gpt']
        # Get indexes of columns to keep
        include_cols = [col for col in df.columns if col not in exclude_cols]
        columns_to_normalize = [df.columns.get_loc(col) for col in include_cols] 
        target_index = df.columns.to_list().index('Close')
        train1, test1 = train_test_split_time_series(df, test_size=0.3)

    elif dataset == 'goog' or dataset == 'GOOG.csv':
        df = pd.read_csv(r'data\\GOOG.csv')
        df['time_step'] = range(len(df))
        df = df.drop('Date', axis=1)
        df = df.drop('News_flag', axis=1)
        if use_sentiment == 0:
            df = df.drop('Scaled_sentiment', axis=1)
            df = df.drop('Sentiment_gpt', axis=1)
        else:
            df['Scaled_sentiment'] = df['Scaled_sentiment'].shift(use_sentiment).fillna(0)
            df['Sentiment_gpt'] = df['Sentiment_gpt'].shift(use_sentiment).fillna(0)
        exclude_cols = []#['Scaled_sentiment', 'Sentiment_gpt']
        # Get indexes of columns to keep
        include_cols = [col for col in df.columns if col not in exclude_cols]
        columns_to_normalize = [df.columns.get_loc(col) for col in include_cols] 
        target_index = df.columns.to_list().index('Close')
        train1, test1 = train_test_split_time_series(df, test_size=0.3)

    elif dataset == 'fin_wmt' or dataset == 'WMT.csv':
        df = pd.read_csv(r'data\\WMT.csv')
        # target_index = df.columns.to_list().index('Close')
        df['time_step'] = range(len(df))
        df = df.drop('Date', axis=1)
        df = df.drop('News_flag', axis=1)
        if use_sentiment == 0:
            df = df.drop('Scaled_sentiment', axis=1)
            df = df.drop('Sentiment_gpt', axis=1)

        exclude_cols = []#['Scaled_sentiment', 'Sentiment_gpt']
        # Get indexes of columns to keep
        include_cols = [col for col in df.columns if col not in exclude_cols]
        columns_to_normalize = [df.columns.get_loc(col) for col in include_cols] 
        target_index = df.columns.to_list().index('Close')
        train1, test1 = train_test_split_time_series(df, test_size=0.3)
    elif dataset == 'fin_amzn' or dataset == 'AMZN.csv':
        df = pd.read_csv(r'data\\AMZN.csv')
        # target_index = df.columns.to_list().index('Close')
        df['time_step'] = range(len(df))
        df = df.drop('Date', axis=1)
        df = df.drop('News_flag', axis=1)
        if use_sentiment == 0:
            df = df.drop('Scaled_sentiment', axis=1)
            df = df.drop('Sentiment_gpt', axis=1)

        exclude_cols = []#['Scaled_sentiment', 'Sentiment_gpt']
        # Get indexes of columns to keep
        include_cols = [col for col in df.columns if col not in exclude_cols]
        columns_to_normalize = [df.columns.get_loc(col) for col in include_cols] 
        target_index = df.columns.to_list().index('Close')
        train1, test1 = train_test_split_time_series(df, test_size=0.3)
    elif dataset == 'fin_baba' or dataset == 'BABA.csv':
        df = pd.read_csv(r'data\\BABA.csv')
        # target_index = df.columns.to_list().index('Close')
        df['time_step'] = range(len(df))
        df = df.drop('Date', axis=1)
        df = df.drop('News_flag', axis=1)
        if use_sentiment == 0:
            df = df.drop('Scaled_sentiment', axis=1)
            df = df.drop('Sentiment_gpt', axis=1)

        exclude_cols = []#['Scaled_sentiment', 'Sentiment_gpt']
        # Get indexes of columns to keep
        include_cols = [col for col in df.columns if col not in exclude_cols]
        columns_to_normalize = [df.columns.get_loc(col) for col in include_cols] 
        target_index = df.columns.to_list().index('Close')
        train1, test1 = train_test_split_time_series(df, test_size=0.3)
    elif dataset == 'fin_brkb' or dataset == 'BRK-B.csv':
        df = pd.read_csv(r'data\\BRK-B.csv')
        # target_index = df.columns.to_list().index('Close')
        df['time_step'] = range(len(df))
        df = df.drop('Date', axis=1)
        df = df.drop('News_flag', axis=1)
        if use_sentiment == 0:
            df = df.drop('Scaled_sentiment', axis=1)
            df = df.drop('Sentiment_gpt', axis=1)

        exclude_cols = []#['Scaled_sentiment', 'Sentiment_gpt']
        # Get indexes of columns to keep
        include_cols = [col for col in df.columns if col not in exclude_cols]
        columns_to_normalize = [df.columns.get_loc(col) for col in include_cols] 
        target_index = df.columns.to_list().index('Close')
        train1, test1 = train_test_split_time_series(df, test_size=0.3)
    elif dataset == 'fin_cost' or dataset == 'COST.csv':
        df = pd.read_csv(r'data\\COST.csv')
        # target_index = df.columns.to_list().index('Close')
        df['time_step'] = range(len(df))
        df = df.drop('Date', axis=1)
        df = df.drop('News_flag', axis=1)
        if use_sentiment == 0:
            df = df.drop('Scaled_sentiment', axis=1)
            df = df.drop('Sentiment_gpt', axis=1)

        exclude_cols = []#['Scaled_sentiment', 'Sentiment_gpt']
        # Get indexes of columns to keep
        include_cols = [col for col in df.columns if col not in exclude_cols]
        columns_to_normalize = [df.columns.get_loc(col) for col in include_cols] 
        target_index = df.columns.to_list().index('Close')
        train1, test1 = train_test_split_time_series(df, test_size=0.3)
    elif dataset == 'fin_ebay' or dataset == 'ebay.csv':
        df = pd.read_csv(r'data\\ebay.csv')
        # target_index = df.columns.to_list().index('Close')
        df['time_step'] = range(len(df))
        df = df.drop('Date', axis=1)
        df = df.drop('News_flag', axis=1)
        if use_sentiment == 0:
            df = df.drop('Scaled_sentiment', axis=1)
            df = df.drop('Sentiment_gpt', axis=1)

        exclude_cols = []#['Scaled_sentiment', 'Sentiment_gpt']
        # Get indexes of columns to keep
        include_cols = [col for col in df.columns if col not in exclude_cols]
        columns_to_normalize = [df.columns.get_loc(col) for col in include_cols] 
        target_index = df.columns.to_list().index('Close')
        train1, test1 = train_test_split_time_series(df, test_size=0.3)
    # if use_sentiment:
    if preprocess_type == 'decompose':
        output_dim = 2  # trend, seasonal, residual
    else:
        output_dim = 1
    train1, test1, target_index = preprocess(preprocess_type, train1, test1, target_index)   
    if preprocess_type == 'decompose':
        columns_to_normalize = range(len(train1.columns))#[train1.columns.get_loc('trend'), train1.columns.get_loc('seasonal'), train1.columns.get_loc('residual')]
    train_dataset, input_dim, train_dataset_actual = create_seqs_normalized([train1],train1.columns, seq_len, pred_len, normalization, columns_to_normalize, target_index, w_aug)
    test_dataset, input_dim, test_dataset_actual = create_seqs_normalized([test1],test1.columns, seq_len, pred_len, normalization, columns_to_normalize, target_index, w_aug)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader_actual = DataLoader(test_dataset_actual, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    train_loader_actual = DataLoader(train_dataset_actual, batch_size=batch_size, shuffle=True)
    # if eda:
    #     plot_train_test_target_distributions(train_loader, test_loader, num_outputs=len(target_index))
    return train1, test1, train_loader, test_loader, train_loader_actual, test_loader_actual, input_dim, output_dim, train1.columns.tolist(), target_index, columns_to_normalize
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
