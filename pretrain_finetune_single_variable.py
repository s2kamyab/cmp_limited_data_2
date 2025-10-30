#! C:\Users\shima\Documents\Postdoc_Uvic\Paper1\Code\Github\uvic_paper\Scripts\python.exe
import torch
from utils.data_loader_pretrain import load_data_pretrain as load_data
from utils.model_loader import load_model
from utils.training import train_model_only_target
from utils.evaluation import evaluate_model
from utils.EDA import Explore_data
from utils.evaluation import evaluate_forecast
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
import os
from torch.utils.data import ConcatDataset, DataLoader
# from utils.augmentation import *

def load_all_except_one(folder_path, exclude_file, preprocess_type, 
                                                     seq_len, pred_len,batch_size,
                                                     normalization, use_sentiment, w_augment):
    # Step 1: Load the excluded file separately
    excluded_path = exclude_file#os.path.join(folder_path, exclude_file)
    train1_ex, test1_ex, train_loader_ex, test_loader_ex, train_loader_actual_ex ,\
       test_loader_actual_ex , input_dim_ex, output_dim_ex, cols_ex, target_index_ex, \
        columns_to_normalize_ex = load_data(excluded_path, preprocess_type, 
                                             seq_len, pred_len, 
                                             batch_size, normalization, 
                                             use_sentiment, w_augment)

    # Step 2: Initialize accumulators for "rest" data
    all_train_sets = []
    all_test_sets = []
    all_train_sets_actual = []
    all_test_sets_actual = []
    file_list = os.listdir(folder_path)
    # Step 3: Loop through all other files
    for file in file_list:#[:4]:  # Limit to first 3 files for testing
        if file != exclude_file:
            full_path = file#os.path.join(folder_path, file)
            train1, test1, train_loader, test_loader, train_loader_actual, \
                test_loader_actual, input_dim, output_dim, cols, target_index,\
                      columns_to_normalize = load_data(full_path, preprocess_type, 
                                             seq_len, pred_len, 
                                             batch_size, normalization, 
                                             use_sentiment, w_augment)
            all_train_sets_actual.append(train_loader_actual.dataset)
            all_test_sets_actual.append(test_loader_actual.dataset)
            all_train_sets.append(train_loader.dataset)
            all_test_sets.append(test_loader.dataset)
        
    # Step 4: Concatenate all datasets and recreate DataLoaders
    combined_train_loader_actual = DataLoader(ConcatDataset(all_train_sets_actual), batch_size=64, shuffle=True)
    combined_test_loader_actual = DataLoader(ConcatDataset(all_test_sets_actual), batch_size=64, shuffle=False)
    combined_train_loader = DataLoader(ConcatDataset(all_train_sets), batch_size=64, shuffle=True)
    combined_test_loader = DataLoader(ConcatDataset(all_test_sets), batch_size=64, shuffle=False)


    return {
        "excluded": {
            "train1": train1_ex,
            "test1": test1_ex,
            "train_loader": train_loader_ex,
            "test_loader": test_loader_ex,
            "train_loader_actual": train_loader_actual_ex,
            "test_loader_actual": test_loader_actual_ex,
            "input_dim": input_dim_ex,
            "output_dim": output_dim_ex,
            "cols": cols_ex,
            "target_index": target_index_ex,
            "columns_to_normalize": columns_to_normalize_ex,
        },
        "combined_others": {
            "train1": train1,
            "test1": test1,
            "train_loader": combined_train_loader,
            "test_loader": combined_test_loader,
            "train_loader_actual": combined_train_loader_actual,
            "test_loader_actual": combined_test_loader_actual,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "cols": cols,
            "target_index": target_index,
            "columns_to_normalize": columns_to_normalize,
            
        }
    }

def main():
    # Framework Settings
    finetune_dataset_name = 'soshianest_530501'#soshianest_5627','soshianest_530486', 'soshianest_530501', 'soshianest_549324', 
#  'fin_aal', 'fin_aapl', 'fin_amd', 'fin_ko', 'fin_TSM', 'goog', 'fin_wmt', 'fin_amzn', 'fin_baba',
# 'fin_brkb', 'fin_cost', 'fin_ebay', 'clarckson_47353'
    
    normalization = 'standard' ##'relative'#'uniform'# 'standard' # 'None'
    pred_len = 1
    seq_len = 12
    batch_size = 12
    preprocess_type ='None'#'fft'#'decompose'#'None'
    eda = True
    model_type = 'GPT2like_transformer'#'ets'#'GPT2like_transformer'# 'rnn', 'cnn', 'gru', 'finspd_transformer', 'lstm', 'times_net'
    epoch = 10
    lr = 0.00001
    phase = 'train'  # 'train' or 'test
    use_sentiment = 0# 0 --> no sentiment, int --> lagged version of sentiment
    w_augment = {'w_jit': 0, 'w_crop':0, 
                 'w_mag_warp':0, 'w_time_warp':0, 
                 'w_rotation':0, 'w_rand_perm':0,
                 'w_mbb' : 0, 'w_dbn': 0}
    iter = 1
    plot_res = False # If True, plots the results of the evaluation
    criterion = 'mse' # 'smape', 'mse', 'mae', 'mape' # Loss function to use, can be 'mse', 'mae', 'smape', or 'mape'
    # print(f"Running with dataset: {dataset_name},\n model: {model_type},\n preprocess: {preprocess_type}, \n normalization: {normalization},\n sequence length: {seq_len}, \n prediction length: {pred_len},\n batch size: {batch_size}, \n learning rate: {lr},\n phase: {phase}")
    ####################################################################################
    # Load dataset
    result = load_all_except_one('data_pretr_finetune/', finetune_dataset_name , preprocess_type, 
                                                        seq_len, pred_len,batch_size,
                                                        normalization, use_sentiment, w_augment)
        # Access:
    
    excluded_train_loader = result["excluded"]["train_loader"]
    combined_train_loader = result["combined_others"]["train_loader"]
    excluded_test_loader = result["excluded"]["test_loader"]
    combined_test_loader = result["combined_others"]["test_loader"]
    excluded_train_loader_actual = result["excluded"]["train_loader_actual"]
    combined_train_loader_actual = result["combined_others"]["train_loader_actual"]
    excluded_test_loader_actual = result["excluded"]["test_loader_actual"]
    combined_test_loader_actual = result["combined_others"]["test_loader_actual"]
    input_dim = result["excluded"]["input_dim"]
    output_dim = result["excluded"]["output_dim"]
    cols = result["excluded"]["cols"]
    target_index_ex = result["excluded"]["target_index"]
    target_index = result["combined_others"]["target_index"]
    columns_to_normalize = result["excluded"]["columns_to_normalize"]
    
    #####################################################################################
    # Load model to pre train
    model, optimizer = load_model(model_type, 1, output_dim, seq_len, pred_len, lr)# input dim is 1 for single variable fine tuning
    ##########################################################################################
    chkpnt_path = f'{finetune_dataset_name}_{model_type}_preprocess_{preprocess_type}_normalization_{normalization}_seq_len_{seq_len}_pred_len_{pred_len}_batch_size_{batch_size}_lr_{lr}.pth'  # Ensure the checkpoint path has the correct extension
    print(f"Model {model_type} loaded successfully with input dimension 1 .")
    # chkpnt_path = f'training_results/{dataset_name}_{model_type}_preprocess_{preprocess_type}_normalization_{normalization}_seq_len_{seq_len}_pred_len_{pred_len}_batch_size_{batch_size}_lr_{lr}.pth'  # Ensure the checkpoint path has the correct extension
    # print(f"Model {model_type} loaded successfully with input dimension {input_dim}.")
    ##########################################################################################
        
    # Train model
    mets =[]
    for i in range(iter):
        # Train the model
        train_losses, val_losses, model = train_model_only_target(model, 
                                                            finetune_dataset_name,
                                                            seq_len,
                                                            batch_size,
                                                            model_type,
                                                            combined_train_loader_actual, 
                                                            combined_test_loader_actual, 
                                                            criterion, 
                                                            optimizer, 
                                                            chkpnt_path,
                                                            target_index,
                                                            normalization=normalization, 
                                                            epochs=epoch, 
                                                            device='cpu', 
                                                            load_checkpoint=False)
        # Freeze the first N layers
        N = 1 if model_type in ['times_net'] else 3
        layer_count = 0

        for name, child in model.named_children():
            if layer_count < N:
                for param in child.parameters():
                    param.requires_grad = False
            layer_count += 1


        train_losses, val_losses, model = train_model_only_target(model, 
                                                            finetune_dataset_name,
                                                            seq_len,
                                                            batch_size,
                                                            model_type,
                                                            excluded_train_loader_actual, 
                                                            excluded_test_loader_actual, 
                                                            criterion, 
                                                            optimizer, 
                                                            chkpnt_path,
                                                            target_index,
                                                            normalization=normalization, 
                                                            epochs=epoch, 
                                                            device='cpu', 
                                                            load_checkpoint=False)
        
        print(f"Model {model_type} trained successfully with {len(train_losses)} epochs.")
        ##########################################################################################
        
        # Evaluate model on test data
        mets.append(evaluate_model(model, excluded_test_loader, excluded_test_loader_actual, cols, 
                    train_losses, val_losses, pred_len, normalization, columns_to_normalize,
                    0, preprocess_type, plot_res, num_samples=3, device='cpu'))
    # Initialize sum dictionary
    sum_metrics = defaultdict(float)
    for m in mets:
        for key, value in m.items():
            sum_metrics[ key] += value
    # Compute averages
    n = len(mets)
    avg_metrics = {key: val / n for key, val in sum_metrics.items()}

    print(avg_metrics)
        

if __name__ == '__main__':
    main()
