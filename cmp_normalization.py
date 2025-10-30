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
    dataset_name = 'goog'#'soshianest_5627','soshianest_530486', 'soshianest_530501', 'soshianest_549324', 
#  'fin_aal', 'fin_aapl', 'fin_amd', 'fin_ko', 'fin_TSM', 'goog', 'fin_wmt', 'fin_amzn', 'fin_baba',
# 'fin_brkb', 'fin_cost', 'fin_ebay', 'clarckson_47353', 'clarckson_541976', 'clarckson_42930', 'clarckson_95900'
    # dataset_name = 'clarckson_pca_541976'
    # dataset_name = 'clarckson_pca_95900'  
    # dataset_name = 'clarckson_pca_42930'
    # dataset_name = 'clarckson_pca_47353'
    normalization = ['standard', 'relative', 'minmax' ] ##'relative'#'uniform'# 'standard' # 'None'
    pred_len = 2
    seq_len = 16
    batch_size = 8
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


    
    
    X_all_used_train = train1.copy()#X_all_train.copy()
    X_all_used_test = test1.copy()#X_all_test.copy()


    #####################################################################################
    
    ##########################################################################################
    # Load model
    model, optimizer = load_model(model_type, input_dim,output_dim, seq_len, pred_len, lr)
    # target_index = cols.index(target)  # Update target index based on selected features
    preds_standard, gts_standard = model.forward(pd.DataFrame(X_all_used_train), pd.DataFrame(X_all_used_test), "standard", target_index)
    
    # target_index = selected_features.index(target)  # Update target index based on selected features
    model, optimizer = load_model(model_type, X_all_used_train.shape[1],output_dim, seq_len, pred_len, lr)
    preds_relative, gts_relative = model.forward(pd.DataFrame(X_all_used_train), pd.DataFrame(X_all_used_test), "relative", target_index)

    # target_index = -1#selected_features.index(target)  # Update target index based on selected features
    model, optimizer = load_model(model_type, X_all_used_train.shape[1],output_dim, seq_len, pred_len, lr)
    preds_minmax, gts_minmax = model.forward(pd.DataFrame(X_all_used_train), pd.DataFrame(X_all_used_test), "minmax", target_index)
    # if len(preds.shape) == 2:
    #     preds = np.expand_dims(preds, axis = 2)
    # if len(gts.shape) == 2:
    #     gts = np.expand_dims(gts, axis = 2)

    # Evaluate
    metrics_standard = evaluate_forecast(gts_standard[:,0], preds_standard[:,0])
    metrics_relative = evaluate_forecast(gts_relative[:,0], preds_relative[:,0])
    metrics_minmax = evaluate_forecast(gts_minmax[:,0], preds_minmax[:,0])
    print(metrics_standard)
    print(metrics_relative)
    print(metrics_minmax)


if __name__ == '__main__':
    main()
