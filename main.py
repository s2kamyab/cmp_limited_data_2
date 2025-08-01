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


# from utils.augmentation import *



def main():
    # Framework Settings
    dataset_name = 'fin_aal'#'soshianest_530486', 'soshianest_530501', 'soshianest_549324', 
#  'fin_aal', 'fin_aapl', 'fin_amd', 'fin_ko', 'fin_TSM', 'goog', 'fin_wmt', 'fin_amzn', 'fin_baba',
# 'fin_brkb', 'fin_cost', 'fin_ebay', 'clarckson_47353', 'clarckson_541976', 'clarckson_42930', 'clarckson_95900'
    # dataset_name = 'clarckson_pca_541976'
    # dataset_name = 'clarckson_pca_95900'  
    # dataset_name = 'clarckson_pca_42930'
    # dataset_name = 'clarckson_pca_47353'
    normalization = 'standard' ##'relative'#'uniform'# 'standard' # 'None'
    pred_len = 1
    seq_len = 12
    batch_size = 12
    preprocess_type ='None'#'fft'#'decompose'#'None'
    eda = True
    model_type = 'pretrained_gpt2'#'ets'#'GPT2like_transformer'# 'rnn', 'cnn', 'gru', 
    # 'finspd_transformer', 'lstm', 'times_net', 'pretrained_gpt2', 'pretrained_autoformer', 'pretrained_informer', 'var'
    # pcs = 1 # Number of principal components to use, 0 means no PCA
    epoch = 100
    lr = 0.00001
    phase = 'train'  # 'train' or 'test
    
    use_sentiment = 0# 0 --> no sentiment, int --> lagged version of sentiment
    w_augment = {'w_jit': 0, 'w_crop':0, 
                 'w_mag_warp':0, 'w_time_warp':0, 
                 'w_rotation':0, 'w_rand_perm':0,
                 'w_mbb' : 0, 'w_dbn': 0}
    iter = 5
    plot_res = False# If True, plots the results of the evaluation
    criterion = 'mse' # 'smape', 'mse', 'mae', 'mape' # Loss function to use, can be 'mse', 'mae', 'smape', or 'mape'
    print(f"Running with dataset: {dataset_name},\n model: {model_type},\n preprocess: {preprocess_type}, \n normalization: {normalization},\n sequence length: {seq_len}, \n prediction length: {pred_len},\n batch size: {batch_size}, \n learning rate: {lr},\n phase: {phase}")
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
    #####################################################################################
    # Explore data
    Explore_data(eda, train_loader, test_loader, preprocess_type, cols, dataset_name, use_sentiment)
    #####################################################################################
    # Load model
    model, optimizer = load_model(model_type, input_dim,output_dim, seq_len, pred_len, lr)
    chkpnt_path = f'training_results/{dataset_name}_{model_type}_preprocess_{preprocess_type}_normalization_{normalization}_seq_len_{seq_len}_pred_len_{pred_len}_batch_size_{batch_size}_lr_{lr}.pth'  # Ensure the checkpoint path has the correct extension
    print(f"Model {model_type} loaded successfully with input dimension {input_dim}.")
    ##########################################################################################
    if model_type not in {'var', 'ets'} :
        
        # Train model
        if phase == 'train':
            mets =[]
            for i in range(iter):
                # Train the model
                train_losses, val_losses, model = train_model(model, 
                                                            train_loader_actual, 
                                                            test_loader_actual, 
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
                mets.append(evaluate_model(model, test_loader, test_loader_actual, cols, 
                            train_losses, val_losses, pred_len, normalization, columns_to_normalize,
                            target_index, preprocess_type, plot_res, num_samples=3, device='cpu'))
            # Initialize sum dictionary
            sum_metrics = defaultdict(float)
            for m in mets:
                for key, value in m.items():
                    sum_metrics[ key] += value
            # Compute averages
            n = len(mets)
            avg_metrics = {key: val / n for key, val in sum_metrics.items()}

            print(avg_metrics)
        elif phase =='test':
            # Load the model from checkpoint
            model, optimizer, epoch, train_losses, val_losses = load_checkpoint_me(model, optimizer, path=chkpnt_path)
            print(f"Model {model_type} loaded from checkpoint {chkpnt_path}.")
            ##########################################################################################
            # Evaluate model on test data
            evaluate_model(model, test_loader, test_loader_actual, cols, 
                        train_losses, val_losses, pred_len, normalization, columns_to_normalize,
                        target_index, preprocess_type, plot_res,  num_samples=3, device='cpu')
    else:
        preds, gts = model.forward(train1, test1, normalization, target_index)
        if len(preds.shape) == 2:
            preds = np.expand_dims(preds, axis = 2)
        if len(gts.shape) == 2:
            gts = np.expand_dims(gts, axis = 2)
        if plot_res == True:
            
            # for i in range(7):
            #     plt.figure()
            #     plt.plot(range(preds.shape[1]),preds[i, :,0],'-.r', gts[i, :,0] , '-.b')
            plt.figure()
            plt.plot(range(preds.shape[0]), preds[:,0,0], gts[:,0,0])
            plt.grid()
            

        # Evaluate
        metrics = evaluate_forecast(gts[:,0,0], preds[:,0,0])
        # metrics = [metrics]
        # for k in range(preds.shape[0]):
            # metrics.append(evaluate_forecast(gts[k], preds[k]))
        # Initialize sum dictionary
        # sum_metrics = defaultdict(float)
        # for m in metrics:
        #     for key, value in m.items():
        #         sum_metrics[ key] += value
        # # Compute averages
        # n = len(metrics)
        # avg_metrics = {key: val / n for key, val in sum_metrics.items()}

        # print(avg_metrics)
        print(metrics)
        plt.show()

if __name__ == '__main__':
    main()
