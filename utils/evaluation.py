import matplotlib.pyplot as plt 
import numpy as np
import torch
from data_loader import normalise_selected_columns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def smape(y_true, y_pred):
    eps=0.1
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Avoid divide-by-zero
    denominator = np.where(denominator == 0, 1e-8, denominator)
    mask = denominator > eps  # avoid division by small number
    smape = np.mean(np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100
    return smape#np.mean(np.abs(y_pred - y_true) / denominator) * 100

def mape(y_true, y_pred):
    # Avoid divide-by-zero
    eps=0.1
    
    y_true = np.where(y_true == 0, 1e-8, y_true)
    mask = y_true > eps 
    return np.mean(np.abs((y_true [mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_forecast(y_true, y_pred):
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'SMAPE (%)': smape(y_true, y_pred),
        'MAPE (%)': mape(y_true, y_pred)
    }
    return metrics
############################################################################################
def plot_sample_predictions(model,
                            common_cols, 
                            pred_len, 
                            test_loader_actual, 
                            normalization,
                            columns_to_normalize, 
                            target_index, 
                            preprocess,
                            num_samples=3, 
                            device='cpu'):
    model.eval()

    with torch.no_grad():
       
        for i in range(5):#len(y) ):
            x, y = next(iter(test_loader_actual))
            x = x.to(device).numpy()
            y = y.to(device).numpy()
            if preprocess == 'decomposed':
                target_index1 = common_cols.index('trend')
                # target_index2 = common_cols.index('residual')
                target_index3 = common_cols.index('seasonal')
                target_indexes = [target_index1,  target_index3]#[target_index1, target_index2, target_index3]
            else:
                target_indexes = [target_index]
            x_seq = x[i].copy()
            # Calculate causal stats up to t (inclusive)
            if normalization == 'standard':
                means = np.mean(x_seq[:, columns_to_normalize], axis=0)#.mean()
                stds = np.std(x_seq[:, columns_to_normalize], axis=0)#.std().replace(0, 1e-8)
                
                means = np.expand_dims(means, axis=0)
                stds = np.expand_dims(stds, axis=0)
                stds[stds == 0] = 1e-8
                # Normalize past values (including target at time t)
                normalized = x_seq
                normalized[:, columns_to_normalize] = (x_seq[:, columns_to_normalize] - np.tile(means, [x_seq.shape[0], 1])) / np.tile(stds, [x_seq.shape[0], 1])
                normalized = np.nan_to_num(normalized, nan=0)
            elif normalization == 'uniform':
                mins = np.min(x_seq[:, columns_to_normalize], axis=0)#.mean()
                maxs = np.max(x_seq[:, columns_to_normalize], axis=0)#.std().replace(0, 1e-8)
                
                mins = np.expand_dims(mins, axis=0)
                maxs = np.expand_dims(maxs, axis=0)
                # Normalize past values (including target at time t)
                normalized = x_seq
                normalized[:, columns_to_normalize] = (x_seq[:, columns_to_normalize] - np.tile(mins, [x_seq.shape[0], 1])) / np.tile(maxs-mins, [x_seq.shape[0], 1])
                # normalized.fillna(0, inplace=True)  # Fill NaN values with 0 after normalization
                normalized = np.nan_to_num(normalized, nan=0)
            elif normalization == 'relative':
                normalized = normalise_selected_columns(x_seq, columns_to_normalize, single_window=True)
                normalized = np.nan_to_num(normalized, nan=0)
            elif normalization == 'None':
                normalized = x_seq
                
            pred = model(torch.unsqueeze(torch.tensor(normalized).float(), dim=0))

            pred = pred.squeeze()

            if len(pred.shape) == 1:
                pred = torch.unsqueeze(pred, dim=1) 

            t = pred.cpu().numpy()
            if t.shape[1] > 1:
                t = np.sum(t, axis=1)  # Sum across the last dimension if multiple targets           
            if normalization == 'standard':
                # De-normalize the predictions
                pred_actual = t * np.sum(stds[0, target_indexes], axis = 1) + np.sum(means[0, target_indexes], axis = 1)
            elif normalization == 'uniform':
                if len(target_indexes) > 1:
                    pred_actual = t * np.sum(maxs[0, target_indexes] - mins[0, target_indexes], axis = 1) + np.sum(mins[0, target_indexes], axis = 1)   
                else:
                    pred_actual = t * (maxs[0, target_index] - mins[0, target_index]) + mins[0, target_index] 
                # pred_actual = t * (tt - np.sum(mins[0, target_indexes], axis = 1)) + np.sum(mins[0, target_indexes], axis = 1)
            elif normalization == 'relative':
                if len(target_indexes) > 1:
                    pred_actual = t * np.sum(x_seq[-1, target_indexes], axis = 1)
                else:
                    pred_actual = t * x_seq[-1, target_index]
                
            elif normalization == 'None':
                pred_actual = t

            # Combine predictions
            y_pred_total = pred_actual# np.sum(pred_actual, axis=1)#pred_actual_trend + pred_actual_seasonal + pred_actual_resid
            if len(y_pred_total.shape) == 1:
                y_pred_total = np.expand_dims(y_pred_total, axis=1)
            # target_index = common_cols.index('OT')
            x_batch_total =  x[i].copy() 
            y_gt_total = np.sum(y[i,:,:], axis=1)#y[i,:,0] + y[i,:,1] + y[i,:, 2]
            if len(y_gt_total.shape) == 1:
                y_gt_total = np.expand_dims(y_gt_total, axis=1)

            x_hist = x_batch_total[:, target_index]#.cpu().numpy()
            if x_hist.shape[1] > 1:
                x_hist = np.sum(x_hist, axis=1)
            y_true = np.array(y_gt_total)#.item()
                        
            # else:
            y_hat = np.array(y_pred_total)#.item()

            # Plot
            plt.figure(figsize=(6, 3))
            plt.plot(range(len(x_hist)), x_hist, label='History (Target Feature)', marker='o')
            plt.plot(range(len(x_hist), len(x_hist)+pred_len) , y_true, '-go', label='True Next', markersize=8)
            plt.plot(range(len(x_hist), len(x_hist)+pred_len) , y_hat, '-rx', label='Predicted Next', markersize=8)
            plt.title(f"Sample {i}")
            plt.xlabel("Time Step")
            plt.ylabel("Target Value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"test_sample_prediction{i}.png")



def plot_training_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def evaluate_model(model, 
                   test_loader, 
                   test_loader_actual, 
                   common_cols, 
                   train_losses, 
                   val_losses, 
                   pred_len, 
                   normalization,
                   column_to_normalize, 
                   target_index, 
                   preprocess,
                   plot_res, 
                   num_samples=3, 
                   device='cpu'):

    # Assume `model` and `test_dataloader` are already defined
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader_actual:
            inputs, targets = batch  # Modify if using dict or named tuple
            inputs = inputs.to(device)
            targets = targets.to(device)
            if normalization == 'None':
                inputs_norm = inputs.float()
            elif normalization == 'standard':
                inputs_norm = inputs.float()
                means = inputs.mean(dim=1, keepdim=True)
                stds = inputs.std(dim=1, keepdim=True)#.replace(0, 1e-8)
                stds[stds == 0] = 1e-8  # Avoid division by zero
                inputs_norm = (inputs_norm - means) / stds
            elif normalization == 'uniform':
                inputs_norm = inputs.float()
                mins = inputs.min(dim=1, keepdim=True).values
                maxs = inputs.max(dim=1, keepdim=True).values
                inputs_norm = (inputs_norm - mins) / (maxs - mins)
            elif normalization == 'relative':
                inputs_norm = normalise_selected_columns(inputs, column_to_normalize, single_window=False)
                inputs_norm = torch.tensor(inputs_norm, dtype=torch.float32)  


            if len(inputs_norm.shape) == 2:
                inputs_norm = inputs_norm.unsqueeze(0)

            outputs_norm = model(inputs_norm)
            
            if len(outputs_norm.shape) == 2:
                outputs_norm = torch.unsqueeze(outputs_norm , dim = 2)
            t = outputs_norm
            if normalization == 'standard':
                # means = inputs.mean(dim=1, keepdim=True)
                # stds = inputs.std(dim=1, keepdim=True).replace(0, 1e-8)
                outputs = t * stds[:,:,target_index] + means[:,:,target_index]
            elif normalization == 'uniform':
                # mins = inputs.min(dim=1, keepdim=True).values
                # maxs = inputs.max(dim=1, keepdim=True).values
                outputs = t * (maxs[:,:,target_index] - mins[:,:,target_index]) + mins[:,:,target_index]
            elif normalization == 'relative':
                outputs = t * torch.tile(inputs[:, -1, target_index].unsqueeze(1) , [1,pred_len,1])
            outputs = outputs.cpu().numpy()
            targets = targets.cpu().numpy()
            if len(outputs.shape) == 3:
                outputs = np.sum(outputs, axis=-1)  
                targets = np.sum(targets, axis=-1)  
            all_preds.append(outputs)
            all_targets.append(targets)

    # Concatenate all batches
    y_pred = np.concatenate(all_preds, axis=0).squeeze()
    y_test = np.concatenate(all_targets, axis=0).squeeze()

    # Evaluate
    metrics = evaluate_forecast(y_test, y_pred)
    # for metric, value in metrics.items():
    #     print(f"{metric}: {value:.4f}")
    
    
    if plot_res:

        plot_training_curves(train_losses, val_losses)
        # plt.show()

        plot_sample_predictions(model,
                                common_cols, 
                                pred_len, 
                                test_loader_actual, 
                                normalization, 
                                column_to_normalize, 
                                target_index, 
                                preprocess,
                                num_samples=num_samples, 
                                device=device)

        plt.show()
    return metrics