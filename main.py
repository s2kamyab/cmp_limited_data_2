#! C:\Users\shima\Documents\Postdoc_Uvic\Paper1\Code\Github\uvic_paper\Scripts\python.exe
import torch
from utils.data_loader import load_data
from utils.model_loader import load_model
from utils.training import train_model
from utils.evaluation import evaluate_model
from utils.EDA import Explore_data



def main():
    # Framework Settings
    dataset_name = 'soshianest_5627'#'fin_aal'#'sp500'# 'soshianest_5627' # 'soshianest_530486', 'soshianest_530501', 'soshianest_549324', 'fin_aal', 'fin_aapl', 'fin_abbv', 'fin_amd', 'fin_ko', 'fin_TSM', 'goog', 'fin_wmt'
    # 'soshianest_5627', 'soshianest_530486', 'soshianest_530501', 'soshianest_549324', 'fin_aal', 'fin_aapl', 'fin_abbv', 'fin_amd', 'fin_ko', 'fin_TSM', 'goog', 'fin_wmt'
    normalization = 'uniform'
    pred_len = 4
    seq_len = 16
    batch_size = 16
    preprocess_type ='decompose'# 'None'#'decompose'
    eda = False
    model_type = 'GPT2like_transformer'# 'rnn', 'cnn', 'gru', 'finspd_transformer', 'lstm', 'times_net'
    eda = True
    ####################################################################################
    # Load dataset
    train_loader, test_loader, train_loader_actual, test_loader_actual, input_dim, cols= load_data(dataset_name, preprocess_type, seq_len, pred_len,batch_size, normalization)
    #####################################################################################
    # Explore data
    Explore_data(eda, train_loader, test_loader, preprocess_type, cols)
    ##########3##############################################################################
    # Load model
    model, criterion, optimizer, chkpnt_path = load_model(model_type, input_dim, seq_len, pred_len)
    print(f"Model {model_type} loaded successfully with input dimension {input_dim}.")
    ##########################################################################################
    # Train model
    train_losses, val_losses, trained_model = train_model(model, train_loader, test_loader, criterion, optimizer, chkpnt_path, epochs=100, device='cpu', load_checkpoint=False)
    print(f"Model {model_type} trained successfully with {len(train_losses)} epochs.")
    ##########################################################################################
    # Evaluate model on test data
    evaluate_model(trained_model, test_loader_actual, cols, train_losses, val_losses, seq_len)


if __name__ == '__main__':
    main()
