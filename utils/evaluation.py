import matplotlib.pyplot as plt 
import numpy as np
import torch
def plot_sample_predictions_decomposed(model_one4all, common_cols, pred_len,
                                        seq_len, 
                                          test_loader_actual,
                                            num_samples=3, device='cpu'):
    model_one4all.eval()
    
    with torch.no_grad():
        # for x_batch, y_batch in test_loader:
        for x,y in test_loader_actual :#zip(
        #test_loader_trend_actual, test_loader_seasonal_actual, test_loader_residual_actual):
            # x_batch_res, y_batch_res = next(iter(test_loader_residual))
            x = x.to(device).numpy()
            y = y.to(device).numpy()
            

            pred_actual_list = []
            gt_actual_list = []
            # plotted = False
            x_actual_list = []
            for i in range(5):#len(y) ):
                target_index1 = common_cols.index('trend')
                target_index2 = common_cols.index('residual')
                target_index3 = common_cols.index('seasonal')
                x_seq = x[i]#, j:j+seq_len]
                # Calculate causal stats up to t (inclusive)
                means = np.mean(x_seq, axis=0)#.mean()
                stds = np.std(x_seq, axis=0)#.std().replace(0, 1e-8)
                
                means = np.expand_dims(means, axis=0)
                stds = np.expand_dims(stds, axis=0)
                stds[stds == 0] = 1e-8
                # Normalize past values (including target at time t)
                normalized = (x_seq - np.tile(means, [x_seq.shape[0], 1])) / np.tile(stds, [x_seq.shape[0], 1])
                # normalized.fillna(0, inplace=True)  # Fill NaN values with 0 after normalization
                normalized = np.nan_to_num(normalized, nan=0)
                pred = model_one4all(torch.unsqueeze(torch.tensor(normalized) , dim = 0))
                pred = pred.squeeze()
                y_pred_total = []
                x_batch_total = []
                y_gt_total = []
                # for k in range(pred.shape[0]):
                t= pred[:,0].cpu().numpy()
                t = np.squeeze(t)
                pred_actual_trend = t * stds[0, target_index1] + means[0, target_index1]
                
                t = pred[:,1].cpu().numpy()
                t = np.squeeze(t)
                pred_actual_resid = t * stds[0, target_index2] + means[0, target_index2]

                t = pred[:,2].cpu().numpy()
                t = np.squeeze(t)
                pred_actual_seasonal = t * stds[0, target_index3] + means[0, target_index3]

            
            
                # Combine predictions
                y_pred_total = pred_actual_trend + pred_actual_seasonal + pred_actual_resid
                target_index = common_cols.index('OT')
                x_batch_total =  x[i] 
                y_gt_total = y[i,:,0] + y[i,:,1] + y[i,:, 2]

                pred_actual_list.append(y_pred_total)
                gt_actual_list.append(y_gt_total)
                x_actual_list.append(x_batch_total)
                
            for i in range(5):#min(num_samples, len(pred_actual_list))):
                if i > num_samples:
                    # plotted = True
                    return
                else:
                    target_index = common_cols.index('OT')
                    x_hist = x_batch_total[:, target_index]
                    x_hist = x_actual_list[i][:, target_index]#.cpu().numpy()
                    y_true = np.array(gt_actual_list[i])#.item()
                    y_hat = np.array(pred_actual_list[i])#.item()

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


def evaluate_model(trained_model, test_loader_actual, common_cols, train_losses, val_losses, seq_len, num_samples=3, device='cpu'):

    plot_training_curves(train_losses, val_losses)

    plot_sample_predictions_decomposed(trained_model,
                                        common_cols, seq_len, test_loader_actual,
                                        num_samples=num_samples, device=device)
    plt.show()