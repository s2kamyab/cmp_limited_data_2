import torch
from utils.data_loader import load_data
from utils.model_loader import load_model
from utils.training import train_model
from utils.evaluation import evaluate_model



def main():
    # Framework Settings
    dataset_name = 'sp500'# 'clarkson' 
    normalization = 'uniform'
    pred_len = 4
    seq_len = 16
    batch_size = 16
    preprocess_type = 'decompose'
    eda = False
    model_type = 'GPT2like_transformer'# 'rnn', 'cnn', 'gru', 'finspd_transformer', 'lstm', 'times_net'
    # Load dataset
    train_loader, test_loader, train_loader_actual, test_loader_actual, input_dim= load_data(dataset_name, preprocess_type, seq_len, pred_len,batch_size, normalization, eda)
    
    # Load model
    model, criterion, optimizer = load_model(model_type, input_dim, seq_len, pred_len)

    # Train model
    train_losses, val_losses, trained_model = train_model(model, train_loader, test_loader, criterion, optimizer, plot = True)

    # Evaluate model on test data
    evaluate_model(trained_model, test_loader_actual, train_losses, val_losses)


if __name__ == '__main__':
    main()
