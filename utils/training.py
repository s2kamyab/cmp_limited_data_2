import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_loader import HuggingFaceInformerWrapper
import pandas as pd
import numpy as np


class SMAPELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(SMAPELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        numerator = torch.abs(y_pred - y_true)
        denominator = (torch.abs(y_pred) + torch.abs(y_true)) / 2.0
        denominator = torch.where(denominator == 0, torch.tensor(self.epsilon, device=denominator.device), denominator)
        smape = numerator / denominator
        return torch.mean(smape) * 100
    
class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs(y_pred - y_true))


class MAPELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(MAPELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Avoid divide-by-zero
        denominator = torch.where(y_true == 0, torch.tensor(self.epsilon, device=y_true.device), y_true)
        loss = torch.abs((y_true - y_pred) / denominator)
        return torch.mean(loss) * 100

class combined_loss(nn.Module):
    def __init__(self):
        super(combined_loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.smape_loss = SMAPELoss()
        self.mae_loss = MAELoss()
        self.mape_loss = MAPELoss()

    def forward(self, y_pred, y_true):
        mse_loss = self.mse_loss(y_pred, y_true)
        smape_loss = self.smape_loss(y_pred, y_true)
        mae_loss = self.mae_loss(y_pred, y_true)
        mape_loss = self.mape_loss(y_pred, y_true)
        return 0.3 * mse_loss + 0.3 * smape_loss + 0.4 * mae_loss + 0.2 * mape_loss   

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss  # we want to minimize val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"[EarlyStopping] Triggered after {self.patience} epochs.")
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)




def save_checkpoint(model, optimizer, epoch, loss,val_loss, path='checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'val_loss': val_loss
    }, path)


def load_checkpoint_me(model, optimizer, path='checkpoint.pth'):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))  # or 'cuda'
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # Use strict=False to ignore missing keys
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    val_loss = checkpoint['val_loss']
    # print(f"Loaded checkpoint from {path} at epoch {epoch} with loss {loss:.4f} and validation loss {val_loss:.4f}")
    return model, optimizer, epoch, loss, val_loss

def generate_time_features(dates: pd.DatetimeIndex):
    features = {
        "month": dates.month / 12.0,
        "day": dates.day / 31.0,
        "weekday": dates.weekday / 6.0,
        "day_of_year": dates.dayofyear / 366.0,
    }
    return np.stack(list(features.values()), axis=1)  # shape: (context_length, num_features)


def train_model(model,
                dataset_type,
                seq_len,
                batch_size,
                model_type,
                train_loader_actual, 
                test_loader_actual, 
                criterion, 
                optimizer, 
                chkpnt_path,
                target_index,
                normalization='standard', 
                epochs=100, 
                device='cpu', 
                load_checkpoint=False):
    

        
    if load_checkpoint == True:
        model, optimizer, epoch, loss, val_loss = load_checkpoint_me(model, optimizer, path=chkpnt_path)
        print(f"Loaded checkpoint from {chkpnt_path} at epoch {epoch} with loss {loss:.4f}")
    model.to(device)
    if criterion == 'mse':
        criterion = torch.nn.MSELoss()
    elif criterion == 'smape':
        criterion = SMAPELoss()
    elif criterion == 'mae':
        criterion = MAELoss()
    elif criterion == 'mape':
        criterion = MAPELoss()
    elif criterion == 'combined':
        criterion = combined_loss()
        
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    early_stopping = EarlyStopping(patience=7, path=chkpnt_path)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        sample_num_tr = 0
        sample_num_tst = 0
        for x, y in train_loader_actual:
            x, y = x.to(device), y.to(device)
            # --- Normalize input ---
            if normalization == 'standard':
                x_mean = x.mean(dim=1, keepdim=True)  # (B, 1, F)
                x_std = x.std(dim=1, keepdim=True)
                x_std[x_std == 0] = 1e-8
                x_norm = (x - x_mean) / x_std

                # --- Normalize target to same scale (optional but typical) ---
                y_norm = (y - x_mean[:, :, target_index]) /  x_std[:, :, target_index] # Assuming y relates to 1st feature
            elif normalization == 'minmax':
                x_min = x.min(dim=1, keepdim=True)[0]
                x_min = torch.tile(torch.unsqueeze(x_min[:,  target_index], dim=1), [1,y.shape[1],1] )
                x_max = x.max(dim=1, keepdim=True)[0]
                x_max = torch.tile(torch.unsqueeze(x_max[:,  target_index], dim=1), [1,y.shape[1],1] )
                x_norm = (x - x_min) / (x_max - x_min + 1e-8)
                y_norm = (y - x_min) / (x_max - x_min + 1e-8)
            elif normalization == 'relative':
                ref = x[:, -1, :]  # last time step
                ref[ref == 0] = 1e-8
                x_norm = x / torch.tile(torch.unsqueeze(ref,dim=1),[1,x.shape[1],1])  # assuming x relates to 1st feature
                y_norm = y / torch.tile(torch.unsqueeze(ref[:,  target_index], dim=1), [1,y.shape[1],1] ) # assuming y relates to 1st feature

            # --- Model forward ---
            if model_type.endswith('informer'):
                # Create time features for each batch
                start_date = "2000-01-07"
                if dataset_type.startswith('clarckson'):
                    freq = 'W'
                else:
                    freq = "D"  # Daily data; use "H" for hourly, "M" for monthly, etc.

                # Simulate timestamps
                time_index = pd.date_range(start=start_date, periods=seq_len, freq=freq)
                time_feats = generate_time_features(time_index)  # shape: (context_length, num_features)
                time_feats = np.tile(time_feats, (batch_size, 1, 1))  # shape: (batch, context_length, num_features)

                # Convert to tensor
                past_time_features = torch.tensor(time_feats, dtype=torch.float32)  # shape: (B, L, F)
                past_observed_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
                # past_observed_mask = (~torch.isnan(x_norm)).float()
                output_norm = model(x_norm, past_observed_mask, past_time_features)
            else:
                output_norm = model(x_norm)


            if len(output_norm.shape) == 3:
                if output_norm.shape[2] == 2: # preprocess == decompose
                    output_norm = output_norm.sum(dim=2)  # sum over the last dimension
                    y_norm = y_norm.sum(dim=2)  # sum over the last dimension

            if len(output_norm.shape) == 2:
                output_norm = torch.unsqueeze(output_norm, dim=2)
            if len(y_norm.shape) == 2:
                y_norm = torch.unsqueeze(y_norm, dim=2)
              
            if normalization == 'standard':
                if len(target_index)>1:
                    tt = torch.unsqueeze(x_std[:, :, target_index], dim=2)
                    tt2 = torch.unsqueeze(x_mean[:, :, target_index], dim=2)
                else:
                    tt = x_std[:, :, target_index]
                    tt2 = x_mean[:, :, target_index]
                tt = torch.sum(tt, dim = 2)
                tt = torch.unsqueeze(tt, dim=1)
                tt2 = torch.sum(tt2, dim = 2)
                tt2 = torch.unsqueeze(tt2, dim=1)
                output = output_norm * torch.tile(tt, [1,y.shape[1],1]) + torch.tile(tt2, [1,y.shape[1],1])
            elif normalization == 'minmax':
                if len(target_index)>1:
                    tt = torch.unsqueeze(x_min[:, :, target_index], dim=2)
                    tt2 = torch.unsqueeze(x_max[:, :, target_index], dim=2)
                else:
                    tt = x_min[:, :, target_index]
                    tt2 = x_max[:, :, target_index]
                tt = torch.sum(x_max[:, :, target_index], dim = 2)
                tt = torch.unsqueeze(tt, dim=1)
                tt2 = torch.sum(x_min[:, :, target_index], dim = 2)
                tt2 = torch.unsqueeze(tt2, dim=1)
                output = output_norm * (torch.tile(tt, [1,y.shape[1],1]) - torch.tile(tt2, [1,y.shape[1],1])) + torch.tile(tt2, [1,y.shape[1],1])  
            elif normalization == 'relative':
                # if len(target_index.shape)>1:
                #     tt = torch.unsqueeze(x_std[:, :, target_index], dim=2)
                #     tt2 = torch.unsqueeze(x_mean[:, :, target_index], dim=2)
                # else:
                #     tt = x_std[:, :, target_index]
                #     tt2 = x_mean[:, :, target_index]
                tt = torch.sum(ref[:,  target_index], dim=1)
                if len(tt.shape) == 1:
                    tt = torch.unsqueeze(tt, dim=1)
                tt = torch.unsqueeze(tt, dim=1)
                tt = torch.tile(tt, [1,y.shape[1],1])
                output = output_norm * tt# torch.tile(torch.unsqueeze(tt, dim=1), [1,y.shape[1],1] )


            # if normalization == 'standard':
            #     output = output_norm * x_std[:, :, target_index] + x_mean[:, :, target_index]
            # elif normalization == 'minmax':
            #     output = output_norm * (x_max[:, :, target_index] - x_min[:, :, target_index]) + x_min[:, :, target_index]  
            # elif normalization == 'relative':
            #     output = output_norm * torch.tile(torch.unsqueeze(ref[:,  target_index], dim=1), [1,y.shape[1],1] )
            # # x, y = x.to(device), y.to(device)
            # # # 4. Weighted MSE Loss
            # if len(output.shape) == 3:
            #     if output.shape[2] == 2: # preprocess == decompose
            #         output = output.sum(dim=2)  # sum over the last dimension
            #         y = y.sum(dim=2)  # sum over the last dimension
            # loss = criterion(torch.squeeze(output), torch.squeeze(y))#torch.mean(weights * (model_residual(x) - y)**2)
            loss = criterion(torch.squeeze(output_norm), torch.squeeze(y_norm))#torch.mean(weights * (model_residual(x) - y)**2)

            # Compute MSE per element, no reduction
            # y_pred = torch.squeeze(output)
            # y_true = torch.squeeze(y)
            # mse_per_element = F.mse_loss(y_pred, y_true, reduction='none')  # shape: (B, T, n)

            # # Average over seq_len and feature dims to get per-sample loss
            # loss = mse_per_element.mean(dim=[1, 2], keepdim=True)  # shape: (B, 1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            sample_num_tr += 1
            optimizer.zero_grad() 
        train_losses.append(total_loss / sample_num_tr)#len(train_loader_actual))
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in test_loader_actual:
                x, y = x.to(device), y.to(device)
                # --- Normalize input ---
                if normalization == 'standard':
                    x_mean = x.mean(dim=1, keepdim=True)  # (B, 1, F)
                    x_std = x.std(dim=1, keepdim=True)
                    x_std[x_std == 0] = 1e-8
                    x_norm = (x - x_mean) / x_std

                    # --- Normalize target to same scale (optional but typical) ---
                    y_norm = (y - x_mean[:, :, target_index]) / x_std[:, :, target_index]  # Assuming y relates to 1st feature
                elif normalization == 'minmax':
                    x_min = x.min(dim=1, keepdim=True)[0]
                    x_min = torch.tile(torch.unsqueeze(x_min[:,  target_index], dim=1), [1,y.shape[1],1] )
                    x_max = x.max(dim=1, keepdim=True)[0]
                    x_max = torch.tile(torch.unsqueeze(x_max[:,  target_index], dim=1), [1,y.shape[1],1] )
                    x_norm = (x - x_min) / (x_max - x_min + 1e-8)
                    y_norm = (y - x_min) / (x_max - x_min + 1e-8)
                elif normalization == 'relative':
                    ref = x[:, -1, :]  # last time step
                    ref[ref == 0] = 1e-8
                    x_norm = x / torch.tile(torch.unsqueeze(ref,dim=1),[1,x.shape[1],1])  # assuming x relates to 1st feature
                    y_norm = y / torch.tile(torch.unsqueeze(ref[:,  target_index], dim=1), [1,y.shape[1],1] ) # assuming y relates to 1st feature

                # --- Model forward ---
                
                output_norm = model(x_norm)

                # x, y = x.to(device), y.to(device)
                # # 4. Weighted MSE Loss
                if len(output_norm.shape) == 3:
                    if output_norm.shape[2] == 2: # preprocess == decompose
                        output_norm = output_norm.sum(dim=2)  # sum over the last dimension
                        output_norm = torch.unsqueeze(output_norm, dim=2)
                        y_norm = y_norm.sum(dim=2)  # sum over the last dimension
                        y_norm = torch.unsqueeze(y_norm, dim=2)
                elif len(output_norm.shape) == 2:
                    output_norm = torch.unsqueeze(output_norm, dim=2)


                if normalization == 'standard':
                    tt = torch.sum(x_std[:, :, target_index], dim = 2)
                    tt = torch.tile(torch.unsqueeze(tt, dim=1), [1, y.shape[1], 1])
                    tt2 = torch.sum(x_std[:, :, target_index], dim = 2)
                    tt2 = torch.tile(torch.unsqueeze(tt2, dim=1), [1, y.shape[1], 1])
                    output = output_norm * tt + tt2
                elif normalization == 'minmax':
                    tt = torch.sum(x_max[:, :, target_index], dim = 2)
                    tt = torch.tile(torch.unsqueeze(tt, dim=1), [1, y.shape[1], 1])
                    tt2 = torch.sum(x_min[:, :, target_index], dim = 2)
                    tt2 = torch.tile(torch.unsqueeze(tt2, dim=1), [1, y.shape[1], 1])
                    output = output_norm * (tt - tt2) + tt2  
                elif normalization == 'relative':
                    tt = torch.sum(ref[:,  target_index], dim=1)
                    if len(tt.shape) == 1:
                        tt = torch.unsqueeze(tt, dim=1)

                    tt = torch.unsqueeze(tt, dim=1)
                    tt = torch.tile(tt, [1, y.shape[1], 1])
                    output = output_norm * tt#torch.tile(torch.unsqueeze(torch.sum(ref[:,  target_index], dim=1), dim=1), [1,y.shape[1]] )


                # if normalization == 'standard':
                #     output = output_norm * x_std[:, :, target_index] + x_mean[:, :, target_index]
                # elif normalization == 'minmax':
                #     output = output_norm * (x_max[:, :, target_index] - x_min[:, :, target_index]) + x_min[:, :, target_index]  
                # elif normalization == 'relative':
                #     output = output_norm * torch.tile(torch.unsqueeze(ref[:,  target_index], dim=1), [1,y.shape[1],1] )
                # # x, y = x.to(device), y.to(device)
                # # # 4. Weighted MSE Loss
                # if len(output_norm.shape) == 3:
                #     if output.shape[2] == 2: # preprocess == decompose
                #         output = output.sum(dim=2)  # sum over the last dimension
                #         y = y.sum(dim=2)  # sum over the last dimension

                val_loss += criterion(torch.squeeze(output_norm), torch.squeeze(y_norm))
                # print(val_loss)#torch.mean(weights * (model_residual(x) - y)**2)

                # x, y = x.to(device), y.to(device)    
                # val_loss += criterion(torch.squeeze(model(x)), torch.squeeze(y)).item()
                # Save checkpoint of best model
                # if val_loss < best_val_loss:
                #     best_val_loss = val_loss
                #     save_checkpoint(model, optimizer, epoch, train_losses, path=chkpnt_path)
                sample_num_tst += 1
            val_losses.append(val_loss / sample_num_tst)  # len(test_loader_actual))
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, train_losses, val_losses, path=chkpnt_path)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/sample_num_tr:.4f} | Test Loss: {val_loss/sample_num_tst:.4f}")

        # print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader_actual):.4f} | Test Loss: {val_loss/len(test_loader_actual):.4f}")
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break
    return train_losses, val_losses, model
    # Plot sample predictions
    # load checkpoint
    # model_one4all, optimizer, epoch, loss = load_checkpoint(model_one4all, optimizer, path='C:\\Users\\shima\\Documents\\Postdoc_Uvic\\Paper1\\Code\\Github\\paper_1_git_repo\\training\\checkpoint_one4all_epoch_best.pth')

def train_model_only_target(model,
                dataset_type,
                seq_len,
                batch_size,
                model_type,
                train_loader_actual, 
                test_loader_actual, 
                criterion, 
                optimizer, 
                chkpnt_path,
                target_index,
                normalization='standard', 
                epochs=100, 
                device='cpu', 
                load_checkpoint=False):
    

        
    if load_checkpoint == True:
        model, optimizer, epoch, loss, val_loss = load_checkpoint_me(model, optimizer, path=chkpnt_path)
        print(f"Loaded checkpoint from {chkpnt_path} at epoch {epoch} with loss {loss:.4f}")
    model.to(device)
    if criterion == 'mse':
        criterion = torch.nn.MSELoss()
    elif criterion == 'smape':
        criterion = SMAPELoss()
    elif criterion == 'mae':
        criterion = MAELoss()
    elif criterion == 'mape':
        criterion = MAPELoss()
    elif criterion == 'combined':
        criterion = combined_loss()
        
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    early_stopping = EarlyStopping(patience=7, path=chkpnt_path)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        sample_num_tr = 0
        sample_num_tst = 0
        for x, y in train_loader_actual:
            x, y = x.to(device), y.to(device)
            # x = x[:,:,target_index] ##################only target variable
            # --- Normalize input ---
            if normalization == 'standard':
                x_mean = x.mean(dim=1, keepdim=True)  # (B, 1, F)
                x_std = x.std(dim=1, keepdim=True)
                x_std[x_std == 0] = 1e-8
                x_norm = (x - x_mean) / x_std

                # --- Normalize target to same scale (optional but typical) ---
                y_norm = (y - x_mean) /  x_std # Assuming y relates to 1st feature
            elif normalization == 'minmax':
                x_min = x.min(dim=1, keepdim=True)[0]
                x_min = torch.tile(torch.unsqueeze(x_min, dim=1), [1,y.shape[1],1] )
                x_max = x.max(dim=1, keepdim=True)[0]
                x_max = torch.tile(torch.unsqueeze(x_max, dim=1), [1,y.shape[1],1] )
                x_norm = (x - x_min) / (x_max - x_min + 1e-8)
                y_norm = (y - x_min) / (x_max - x_min + 1e-8)
            elif normalization == 'relative':
                ref = x[:, -1, :]  # last time step
                ref[ref == 0] = 1e-8
                x_norm = x / torch.tile(torch.unsqueeze(ref,dim=1),[1,x.shape[1],1])  # assuming x relates to 1st feature
                y_norm = y / torch.tile(torch.unsqueeze(ref[:,  target_index], dim=1), [1,y.shape[1],1] ) # assuming y relates to 1st feature

            # --- Model forward ---
            if model_type.endswith('informer'):
                # Create time features for each batch
                start_date = "2000-01-07"
                if dataset_type.startswith('clarckson'):
                    freq = 'W'
                else:
                    freq = "D"  # Daily data; use "H" for hourly, "M" for monthly, etc.

                # Simulate timestamps
                time_index = pd.date_range(start=start_date, periods=seq_len, freq=freq)
                time_feats = generate_time_features(time_index)  # shape: (context_length, num_features)
                time_feats = np.tile(time_feats, (batch_size, 1, 1))  # shape: (batch, context_length, num_features)

                # Convert to tensor
                past_time_features = torch.tensor(time_feats, dtype=torch.float32)  # shape: (B, L, F)
                past_observed_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
                # past_observed_mask = (~torch.isnan(x_norm)).float()
                output_norm = model(x_norm, past_observed_mask, past_time_features)
            else:
                output_norm = model(x_norm)


            if len(output_norm.shape) == 3:
                if output_norm.shape[2] == 2: # preprocess == decompose
                    output_norm = output_norm.sum(dim=2)  # sum over the last dimension
                    y_norm = y_norm.sum(dim=2)  # sum over the last dimension

            if len(output_norm.shape) == 2:
                output_norm = torch.unsqueeze(output_norm, dim=2)
            if len(y_norm.shape) == 2:
                y_norm = torch.unsqueeze(y_norm, dim=2)
              
            if normalization == 'standard':
                if len(target_index)>1:
                    tt = torch.unsqueeze(x_std, dim=2)
                    tt2 = torch.unsqueeze(x_mean, dim=2)
                else:
                    tt = x_std
                    tt2 = x_mean
                tt = torch.sum(tt, dim = 2)
                tt = torch.unsqueeze(tt, dim=1)
                tt2 = torch.sum(tt2, dim = 2)
                tt2 = torch.unsqueeze(tt2, dim=1)
                output = output_norm * torch.tile(tt, [1,y.shape[1],1]) + torch.tile(tt2, [1,y.shape[1],1])
            elif normalization == 'minmax':
                if len(target_index)>1:
                    tt = torch.unsqueeze(x_min, dim=2)
                    tt2 = torch.unsqueeze(x_max, dim=2)
                else:
                    tt = x_min
                    tt2 = x_max
                tt = torch.sum(x_max, dim = 2)
                tt = torch.unsqueeze(tt, dim=1)
                tt2 = torch.sum(x_min, dim = 2)
                tt2 = torch.unsqueeze(tt2, dim=1)
                output = output_norm * (torch.tile(tt, [1,y.shape[1],1]) - torch.tile(tt2, [1,y.shape[1],1])) + torch.tile(tt2, [1,y.shape[1],1])  
            elif normalization == 'relative':
                # if len(target_index.shape)>1:
                #     tt = torch.unsqueeze(x_std[:, :, target_index], dim=2)
                #     tt2 = torch.unsqueeze(x_mean[:, :, target_index], dim=2)
                # else:
                #     tt = x_std[:, :, target_index]
                #     tt2 = x_mean[:, :, target_index]
                tt = torch.sum(ref, dim=1)
                if len(tt.shape) == 1:
                    tt = torch.unsqueeze(tt, dim=1)
                tt = torch.unsqueeze(tt, dim=1)
                tt = torch.tile(tt, [1,y.shape[1],1])
                output = output_norm * tt


            loss = criterion(torch.squeeze(output_norm), torch.squeeze(y_norm))#torch.mean(weights * (model_residual(x) - y)**2)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            sample_num_tr += 1
            optimizer.zero_grad() 
        train_losses.append(total_loss / sample_num_tr)#len(train_loader_actual))
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in test_loader_actual:
                x, y = x.to(device), y.to(device)
                # --- Normalize input ---
                if normalization == 'standard':
                    x_mean = x.mean(dim=1, keepdim=True)  # (B, 1, F)
                    x_std = x.std(dim=1, keepdim=True)
                    x_std[x_std == 0] = 1e-8
                    x_norm = (x - x_mean) / x_std

                    # --- Normalize target to same scale (optional but typical) ---
                    y_norm = (y - x_mean) / x_std # Assuming y relates to 1st feature
                elif normalization == 'minmax':
                    x_min = x.min(dim=1, keepdim=True)[0]
                    x_min = torch.tile(torch.unsqueeze(x_min, dim=1), [1,y.shape[1],1] )
                    x_max = x.max(dim=1, keepdim=True)[0]
                    x_max = torch.tile(torch.unsqueeze(x_max, dim=1), [1,y.shape[1],1] )
                    x_norm = (x - x_min) / (x_max - x_min + 1e-8)
                    y_norm = (y - x_min) / (x_max - x_min + 1e-8)
                elif normalization == 'relative':
                    ref = x[:, -1, :]  # last time step
                    ref[ref == 0] = 1e-8
                    x_norm = x / torch.tile(torch.unsqueeze(ref,dim=1),[1,x.shape[1],1])  # assuming x relates to 1st feature
                    y_norm = y / torch.tile(torch.unsqueeze(ref, dim=1), [1,y.shape[1],1] ) # assuming y relates to 1st feature

                # --- Model forward ---
                
                output_norm = model(x_norm)

                # x, y = x.to(device), y.to(device)
                # # 4. Weighted MSE Loss
                if len(output_norm.shape) == 3:
                    if output_norm.shape[2] == 2: # preprocess == decompose
                        output_norm = output_norm.sum(dim=2)  # sum over the last dimension
                        output_norm = torch.unsqueeze(output_norm, dim=2)
                        y_norm = y_norm.sum(dim=2)  # sum over the last dimension
                        y_norm = torch.unsqueeze(y_norm, dim=2)
                elif len(output_norm.shape) == 2:
                    output_norm = torch.unsqueeze(output_norm, dim=2)


                if normalization == 'standard':
                    tt = torch.sum(x_std, dim = 2)
                    tt = torch.tile(torch.unsqueeze(tt, dim=1), [1, y.shape[1], 1])
                    tt2 = torch.sum(x_std, dim = 2)
                    tt2 = torch.tile(torch.unsqueeze(tt2, dim=1), [1, y.shape[1], 1])
                    output = output_norm * tt + tt2
                elif normalization == 'minmax':
                    tt = torch.sum(x_max, dim = 2)
                    tt = torch.tile(torch.unsqueeze(tt, dim=1), [1, y.shape[1], 1])
                    tt2 = torch.sum(x_min, dim = 2)
                    tt2 = torch.tile(torch.unsqueeze(tt2, dim=1), [1, y.shape[1], 1])
                    output = output_norm * (tt - tt2) + tt2  
                elif normalization == 'relative':
                    tt = torch.sum(ref, dim=1)
                    if len(tt.shape) == 1:
                        tt = torch.unsqueeze(tt, dim=1)

                    tt = torch.unsqueeze(tt, dim=1)
                    tt = torch.tile(tt, [1, y.shape[1], 1])
                    output = output_norm * tt#torch.tile(torch.unsqueeze(torch.sum(ref[:,  target_index], dim=1), dim=1), [1,y.shape[1]] )


                # if normalization == 'standard':
                #     output = output_norm * x_std[:, :, target_index] + x_mean[:, :, target_index]
                # elif normalization == 'minmax':
                #     output = output_norm * (x_max[:, :, target_index] - x_min[:, :, target_index]) + x_min[:, :, target_index]  
                # elif normalization == 'relative':
                #     output = output_norm * torch.tile(torch.unsqueeze(ref[:,  target_index], dim=1), [1,y.shape[1],1] )
                # # x, y = x.to(device), y.to(device)
                # # # 4. Weighted MSE Loss
                # if len(output_norm.shape) == 3:
                #     if output.shape[2] == 2: # preprocess == decompose
                #         output = output.sum(dim=2)  # sum over the last dimension
                #         y = y.sum(dim=2)  # sum over the last dimension

                val_loss += criterion(torch.squeeze(output_norm), torch.squeeze(y_norm))
                # print(val_loss)#torch.mean(weights * (model_residual(x) - y)**2)

                # x, y = x.to(device), y.to(device)    
                # val_loss += criterion(torch.squeeze(model(x)), torch.squeeze(y)).item()
                # Save checkpoint of best model
                # if val_loss < best_val_loss:
                #     best_val_loss = val_loss
                #     save_checkpoint(model, optimizer, epoch, train_losses, path=chkpnt_path)
                sample_num_tst += 1
            val_losses.append(val_loss / sample_num_tst)  # len(test_loader_actual))
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, train_losses, val_losses, path=chkpnt_path)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/sample_num_tr:.4f} | Test Loss: {val_loss/sample_num_tst:.4f}")

        # print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader_actual):.4f} | Test Loss: {val_loss/len(test_loader_actual):.4f}")
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break
    return train_losses, val_losses, model
    # Plot sample predictions
    # load checkpoint
    # model_