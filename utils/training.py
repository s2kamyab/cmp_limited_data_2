import torch
def train_model(model, train_loader, test_loader, epochs, criterion, device):
    model.to(device)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model_one4all.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # 4. Weighted MSE Loss
            loss = criterion(torch.squeeze(model_one4all(x)), y)#torch.mean(weights * (model_residual(x) - y)**2)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            optimizer.zero_grad() 

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)    
                val_loss += criterion(torch.squeeze(model_one4all(x)), y).item()
                # Save checkpoint of best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model_one4all, optimizer, epoch, loss, path=r'C:\\Users\shima\\Documents\\Postdoc_Uvic\\Paper1\\Code\\Github\\paper_1_git_repo\\training\\checkpoint_one4all_epoch_best.pth')
        

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Test Loss: {val_loss/len(test_loader):.4f}")

    # Plot sample predictions
    # load checkpoint
    model_one4all, optimizer, epoch, loss = load_checkpoint(model_one4all, optimizer, path='C:\\Users\\shima\\Documents\\Postdoc_Uvic\\Paper1\\Code\\Github\\paper_1_git_repo\\training\\checkpoint_one4all_epoch_best.pth')

