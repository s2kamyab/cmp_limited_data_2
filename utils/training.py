import torch
def save_checkpoint(model, optimizer, epoch, loss, path='checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(model, optimizer, path='checkpoint.pth'):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))  # or 'cuda'
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


def train_model(model, train_loader, test_loader, criterion, optimizer, chkpnt_path, epochs=100, device='cpu', load_checkpoint=False):
    if load_checkpoint == True:
        model, optimizer, epoch, loss = load_checkpoint(model, optimizer, path=chkpnt_path)
        print(f"Loaded checkpoint from {chkpnt_path} at epoch {epoch} with loss {loss:.4f}")
    model.to(device)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # 4. Weighted MSE Loss
            loss = criterion(torch.squeeze(model(x)), y)#torch.mean(weights * (model_residual(x) - y)**2)

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
                val_loss += criterion(torch.squeeze(model(x)), y).item()
                # Save checkpoint of best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, epoch, total_loss/len(train_loader), path=chkpnt_path)
        
        val_losses.append(val_loss / len(test_loader))
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Test Loss: {val_loss/len(test_loader):.4f}")
    return train_losses, val_losses, model
    # Plot sample predictions
    # load checkpoint
    # model_one4all, optimizer, epoch, loss = load_checkpoint(model_one4all, optimizer, path='C:\\Users\\shima\\Documents\\Postdoc_Uvic\\Paper1\\Code\\Github\\paper_1_git_repo\\training\\checkpoint_one4all_epoch_best.pth')

