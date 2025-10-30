#! C:\Users\shima\Documents\Postdoc_Uvic\Paper1\Code\Github\uvic_paper\Scripts\python.exe
import torch
from utils.data_loader import load_data
from utils.model_loader import load_model
from utils.training import train_model, load_checkpoint_me
from utils.evaluation import evaluate_model
from utils.EDA import Explore_data
from utils.evaluation import evaluate_forecast
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# from utils.augmentation import *



def main():
    # Framework Settings
    dataset_name = 'fin_baba'#'soshianest_530486', 'soshianest_530501', 'soshianest_549324', 
#  'fin_aal', 'fin_aapl', 'fin_amd', 'fin_ko', 'fin_TSM', 'goog', 'fin_wmt', 'fin_amzn', 'fin_baba',
# 'fin_brkb', 'fin_cost', 'fin_ebay', 'clarckson_47353', 'clarckson_541976', 'clarckson_42930', 'clarckson_95900'
    # dataset_name = 'clarckson_pca_541976'
    # dataset_name = 'clarckson_pca_95900'  
    # dataset_name = 'clarckson_pca_42930'
    # dataset_name = 'clarckson_pca_47353'
    normalization = 'standard' ##'relative'#'uniform'# 'standard' # 'None'
    pred_len = 2
    seq_len = 12
    batch_size = 12
    preprocess_type ='None'#'fft'#'decompose'#'None'
    model_type = 'var'#'pretrained_informer'#'ets'#'GPT2like_transformer'# 'rnn', 'cnn', 'gru', 
    # 'finspd_transformer', 'lstm', 'times_net', 'pretrained_gpt2', 'pretrained_autoformer', 'pretrained_informer', 'var'
    # pcs = 1 # Number of principal components to use, 0 means no PCA
    epoch = 100
    lr = 0.00001
    phase = 'train'  # 'train' or 'test'
    ####################################################################################
    # Data augmentation weights
    use_sentiment =2# 0 --> no sentiment, int --> lagged version of sentiment
    w_augment = {'w_jit': 0, 'w_crop':0, 
                 'w_mag_warp':0, 'w_time_warp':0, 
                 'w_rotation':0, 'w_rand_perm':0,
                 'w_mbb' : 0, 'w_dbn': 0}
    ####################################################################################
    # Load dataset
    train1, test1,\
    train_loader, test_loader, \
        train_loader_actual, test_loader_actual,\
              input_dim,output_dim, cols, target_index,\
                  columns_to_normalize = load_data(dataset_name, 
                                                   preprocess_type, 
                                                   seq_len, pred_len,batch_size,
                                                     normalization, use_sentiment, 
                                                     w_augment)
    
    ############### important: first run paper_1_git_repo\cross_correlation_multi_variable.py######################3
    # --- Load the feature-flag table (CSV1) ---
    flags_df = pd.read_csv("your_file_with_flag.csv")  # e.g., columns: ["feature_name", "use_flag"]
    # flags_df = flags_df.set_index("above_average")
    # target = "OT"
    # Identify which features to use
    selected_features = flags_df[flags_df["above_average"] == 1]["series"].tolist()
    if cols[target_index[0]] not in selected_features:
        selected_features.append(cols[target_index[0]])  # Ensure target is included
        # target_index = -1  # Update target index to last column
    selected_lags = flags_df[flags_df["above_average"] == 1]["best_lag"].tolist()
    # --- Load the main dataset (CSV2) ---
    data_df = train1#pd.read_csv("paper_1_git_repo/data_soshianest/5627_dataset.csv")  # assume columns matching features + target
    data_df_test = test1
    # Example: assume target column is named "target"
    target = cols[target_index[0]]
    
    X_all_train = data_df#data_df.drop(columns=[target])
    X_all_test = data_df_test#data_df_test.drop(columns=[target])
    # y = data_df[target].values

    # Ensure X_all only contains the feature columns listed in flags
    # tt = #selected_features
    # if 'News_flag'in tt:
    #     tt.remove('News_flag')
    # X_all_train = X_all_train[tt]   # this filters to exactly the features in the flags table
    # X_all_test = X_all_test[tt]
    # # Configuration 1: Selected features only
    # X_sel_train = X_all_train.copy()
    # X_sel_test = X_all_test.copy()
    # for feat, lag in zip(selected_features, selected_lags): # compute lagged versions
    #     if lag > 0:
    #         X_sel_train[feat] = X_all_train[feat].shift(lag)
    #         X_sel_test[feat] = X_all_test[feat].shift(lag)
    
    # X_sel_train = X_sel_train[selected_features].dropna().values  
    # X_sel_test = X_sel_test[selected_features].dropna().values
    X_sel_train = X_all_train[selected_features].values  # not including the target!
    X_sel_test = X_all_test[selected_features].values

    # Configuration 2: All features
    X_all_used_train = X_all_train.copy()
    X_all_used_test = X_all_test.copy()

    # Configuration 3: PCA on all features
    # Split target out
    y_train = X_all_used_train[target].to_numpy()
    y_test  = X_all_used_test[target].to_numpy()

    X_train_feats = X_all_used_train.drop(columns=[target])
    X_test_feats  = X_all_used_test.drop(columns=[target])
    # Keep original target position (index) for reinsertion
    target_idx = X_all_used_train.columns.get_loc(target)

    # Scale + PCA on features only
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train_feats)
    X_scaled_test  = scaler.transform(X_test_feats)

    pca = PCA(n_components=0.95, svd_solver="full")
    X_pca_train1 = pca.fit_transform(X_scaled_train)
    X_pca_test1  = pca.transform(X_scaled_test)
    print("PCA – number of components kept:", pca.n_components_)
    # Reinsert target at original position
    def insert_col_at(df_like_cols, arr_pca, y, idx):
        # Build column names for PCA comps
        pca_cols = [f"pc{i+1}" for i in range(arr_pca.shape[1])]
        # Create DF for PCA features
        df_pca = pd.DataFrame(arr_pca, columns=pca_cols, index=df_like_cols.index)
        # Rebuild full frame with target back at original position
        left  = df_pca.iloc[:, :idx]
        right = df_pca.iloc[:, idx:]
        out = pd.concat([left,
                        pd.Series(y, name=target, index=df_like_cols.index),
                        right],
                        axis=1)
        return out

    X_pca_train = insert_col_at(X_train_feats, X_pca_train1, y_train, target_idx)
    X_pca_test  = insert_col_at(X_test_feats,  X_pca_test1,  y_test,  target_idx)

    # scaler = StandardScaler()
    # X_scaled_train = scaler.fit_transform(X_all_used_train)
    # X_scaled_test = scaler.transform(X_all_used_test)

    # pca = PCA(n_components=0.95, svd_solver="full")  # keep components explaining 95% variance
    # X_pca_train = pca.fit_transform(X_scaled_train)
    # X_pca_test = pca.transform(X_scaled_test)
    # print("PCA – number of components kept:", pca.n_components_)

    #####################################################################################
    
    ##########################################################################################
    # Load model
    model, optimizer = load_model(model_type, input_dim,output_dim, seq_len, pred_len, lr)
    target_index = cols.index(target)  # Update target index based on selected features
    preds_all, gts_all = model.forward(pd.DataFrame(X_all_used_train), pd.DataFrame(X_all_used_test), normalization, target_index)
    
    target_index = selected_features.index(target)  # Update target index based on selected features
    model, optimizer = load_model(model_type, X_sel_train.shape[1],output_dim, seq_len, pred_len, lr)
    preds_sel, gts_sel = model.forward(pd.DataFrame(X_sel_train), pd.DataFrame(X_sel_test), normalization, target_index)
    
    target_index = -1#selected_features.index(target)  # Update target index based on selected features
    model, optimizer = load_model(model_type, X_pca_train.shape[1],output_dim, seq_len, pred_len, lr)
    preds_pca, gts_pca = model.forward(pd.DataFrame(X_pca_train), pd.DataFrame(X_pca_test), normalization, target_index)
    # if len(preds.shape) == 2:
    #     preds = np.expand_dims(preds, axis = 2)
    # if len(gts.shape) == 2:
    #     gts = np.expand_dims(gts, axis = 2)

    # Evaluate
    metrics_all = evaluate_forecast(gts_all[:,0], preds_all[:,0])
    metrics_sel = evaluate_forecast(gts_sel[:,0], preds_sel[:,0])
    metrics_pca = evaluate_forecast(gts_pca[:,0], preds_pca[:,0])
    print(metrics_all)
    print(metrics_sel)
    print(metrics_pca)


if __name__ == '__main__':
    main()
