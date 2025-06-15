import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
import numpy as np
def plot_train_test_target_distributions(train_loader, test_loader,data_name, num_outputs=3):
    # Step 1: Extract targets from DataLoaders
    def extract_targets(loader):
        all_targets = []
        for _, y in loader:
            all_targets.append(y[0])
        return torch.cat(all_targets, dim=0).cpu().numpy()  # shape: (N, num_outputs)

    train_targets = extract_targets(train_loader)
    test_targets = extract_targets(test_loader)
    if train_targets.shape[1] == 3:
        comps = ['trend', 'seasonal', 'residual']
    elif train_targets.shape[1] == 1:
        comps = ['target']

    # Step 2: Plot for each output dimension
    for i in range(num_outputs):
        plt.figure(figsize=(6, 4))
        sns.kdeplot(train_targets[:, i], label='Train', fill=True)
        sns.kdeplot(test_targets[:, i], label='Test', fill=True)
        plt.title(f"Distribution of {comps[i]} for {data_name} Data")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"training_results\\{comps[i]}_{data_name}_distribution.png")
        plt.show()


def compute_feature_correlation(dataloader, features):
    all_data = []

    # Step 1: Accumulate all features from the DataLoader
    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            batch = batch[0]  # ignore labels if present
        # batch shape: (batch_size, seq_len, features)
        all_data.append(batch)

    # Step 2: Concatenate all batches
    data_tensor = np.array(all_data)#torch.cat(all_data, dim=0)  # shape: (total_batches * batch_size, seq_len, features)

    # Step 3: Flatten across time and batch
    B, T, F = data_tensor.shape
    flat_data = data_tensor.reshape(-1, F)#.cpu().numpy()  # shape: (B * T, features)

    # Step 4: Compute correlation matrix using pandas
    df = pd.DataFrame(flat_data[:,:len(features)], columns=features)
    corr_matrix = df.corr()

    return corr_matrix


def Explore_data(eda, train_loader, test_loader, preprocess_type, features, data_name):
    if preprocess_type == 'decompose':
        num_outputs = 3
    else:
        num_outputs = 1
    print(f"Number of outputs: {num_outputs}")

    if eda:
        print("Exploring Training vs Test data distributions...")
        plot_train_test_target_distributions(train_loader, test_loader,data_name, num_outputs=num_outputs)
    
    # if corr:
        print("Calculating and plotting correlation matrix...")
        # Assuming train_loader contains the training data
        train_data = next(iter(train_loader))[0].cpu().numpy()  # Get a batch of data
        corr_matrix = compute_feature_correlation(train_data, features)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title("Correlation Matrix")
        plt.show()
