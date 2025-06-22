#! C:\Users\shima\Documents\Postdoc_Uvic\Paper1\Code\Github\uvic_paper\Scripts\python.exe
import torch
from utils.data_loader import load_data
from utils.model_loader import load_model
from utils.training import train_model, load_checkpoint_me
from utils.evaluation import evaluate_model
from utils.EDA import Explore_data



def main():
    # Framework Settings
    dataset_name = 'fin_aapl'#'soshianest_530486', 'soshianest_530501', 'soshianest_549324', 'fin_aal', 'fin_aapl', 'fin_abbv', 'fin_amd', 'fin_ko', 'fin_TSM', 'goog', 'fin_wmt'
    # 'soshianest_5627', 'soshianest_530486', 'soshianest_530501', 'soshianest_549324', 'fin_aal', 'fin_aapl', 'fin_abbv', 'fin_amd', 'fin_ko', 'fin_TSM', 'goog', 'fin_wmt'
    normalization = 'relative'#'relative'#'uniform'# 'standard' # 'None'
    pred_len = 4
    seq_len = 16
    batch_size = 16
    preprocess_type ='None'#'fft'#'decompose'#'None'#'decompose'# 'None'#'decompose'
    eda = True
    model_type = 'GPT2like_transformer'#'GPT2like_transformer'# 'rnn', 'cnn', 'gru', 'finspd_transformer', 'lstm', 'times_net'
    epoch = 100
    lr = 0.0001
    phase = 'train'  # 'train' or 'test
    use_sentiment = True # Whether to use sentiment data or not
    criterion = 'mse' # 'smape', 'mse', 'mae', 'mape' # Loss function to use, can be 'mse', 'mae', 'smape', or 'mape'
    print(f"Running with dataset: {dataset_name},\n model: {model_type},\n preprocess: {preprocess_type}, \n normalization: {normalization},\n sequence length: {seq_len}, \n prediction length: {pred_len},\n batch size: {batch_size}, \n learning rate: {lr},\n phase: {phase}")
    ####################################################################################
    # Load dataset
    train_loader, test_loader, train_loader_actual, test_loader_actual, input_dim,output_dim, cols, target_index, columns_to_normalize = load_data(dataset_name, preprocess_type, seq_len, pred_len,batch_size, normalization, use_sentiment)
    #####################################################################################
    # Explore data
    Explore_data(eda, train_loader, test_loader, preprocess_type, cols, dataset_name, use_sentiment)
    ##########3##############################################################################
    # Load model
    model, optimizer = load_model(model_type, input_dim,output_dim, seq_len, pred_len, lr)
    chkpnt_path = f'training_results/{dataset_name}_{model_type}_preprocess_{preprocess_type}_normalization_{normalization}_seq_len_{seq_len}_pred_len_{pred_len}_batch_size_{batch_size}_lr_{lr}.pth'  # Ensure the checkpoint path has the correct extension
    print(f"Model {model_type} loaded successfully with input dimension {input_dim}.")
    ##########################################################################################
    # Train model
    if phase == 'train':
        train_losses, val_losses, model = train_model(model, 
                                                    train_loader_actual, 
                                                    test_loader_actual, 
                                                    criterion, 
                                                    optimizer, 
                                                    chkpnt_path,
                                                    target_index,
                                                    normalization='relative', 
                                                    epochs=epoch, 
                                                    device='cpu', 
                                                    load_checkpoint=False)
        
        print(f"Model {model_type} trained successfully with {len(train_losses)} epochs.")
    elif phase =='test':
        # Load the model from checkpoint
        model, optimizer, epoch, train_losses, val_losses = load_checkpoint_me(model, optimizer, path=chkpnt_path)
        print(f"Model {model_type} loaded from checkpoint {chkpnt_path}.")
    ##########################################################################################
    # Evaluate model on test data
    evaluate_model(model, test_loader, test_loader_actual, cols, 
                   train_losses, val_losses, pred_len, normalization, columns_to_normalize,
                   target_index, preprocess_type, num_samples=3, device='cpu')


if __name__ == '__main__':
    main()
