import torch

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


def train_model(model, train_loader, test_loader, criterion, optimizer, chkpnt_path, epochs=100, device='cpu', load_checkpoint=False):
    if load_checkpoint == True:
        model, optimizer, epoch, loss, val_loss = load_checkpoint_me(model, optimizer, path=chkpnt_path)
        print(f"Loaded checkpoint from {chkpnt_path} at epoch {epoch} with loss {loss:.4f}")
    model.to(device)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    early_stopping = EarlyStopping(patience=7, path=chkpnt_path)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # 4. Weighted MSE Loss
            loss = criterion(torch.squeeze(model(x)), torch.squeeze(y))#torch.mean(weights * (model_residual(x) - y)**2)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            optimizer.zero_grad() 
        train_losses.append(total_loss / len(train_loader))
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)    
                val_loss += criterion(torch.squeeze(model(x)), torch.squeeze(y)).item()
                # Save checkpoint of best model
                # if val_loss < best_val_loss:
                #     best_val_loss = val_loss
                #     save_checkpoint(model, optimizer, epoch, train_losses, path=chkpnt_path)
        
        val_losses.append(val_loss / len(test_loader))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, train_losses, val_losses, path=chkpnt_path)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Test Loss: {val_loss/len(test_loader):.4f}")
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break
    return train_losses, val_losses, model
    # Plot sample predictions
    # load checkpoint
    # model_one4all, optimizer, epoch, loss = load_checkpoint(model_one4all, optimizer, path='C:\\Users\\shima\\Documents\\Postdoc_Uvic\\Paper1\\Code\\Github\\paper_1_git_repo\\training\\checkpoint_one4all_epoch_best.pth')

