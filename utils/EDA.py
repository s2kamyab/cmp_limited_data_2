import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from scipy.stats import pearsonr
from scipy.stats import ks_2samp
from scipy.special import rel_entr
def plot_train_test_target_distributions(train_loader, test_loader,data_name, num_outputs=3):
    # Step 1: Extract targets from DataLoaders
    def extract_targets(loader):
        all_targets = []
        for _, y in loader:
            all_targets.append(y[0])
        return torch.cat(all_targets, dim=0).cpu().numpy()  # shape: (N, num_outputs)

    train_targets = extract_targets(train_loader)
    test_targets = extract_targets(test_loader)

    # compute similarity between train and test distributions
    if train_targets.shape[1] == 1:
        tr_counts, tr_bin_edges = np.histogram(train_targets, bins=40, density=True)
        tst_counts, tst_bin_edges = np.histogram(test_targets, bins=40, density=True)
        tr_counts = [tr_counts]
        tst_counts = [tst_counts]
    else:
        # for i in range(train_targets.shape[1]):
        tr_counts = []
        tst_counts = []
        for i in range(train_targets.shape[1]):
            a, b = np.histogram(train_targets[:, i], bins=40, density=True) #for i in range(train_targets.shape[1])], axis=0)
            c, d = np.histogram(test_targets[:, i], bins=40, density=True) #for i in range(test_targets.shape[1])], axis=0)
            tr_counts.append(a)
            tst_counts.append(c)
        # tr_counts = np.array(tr_counts)
        # tst_counts = np.array(tst_counts)


    
    stat, ks_p_value = [], []
    js_similarity = []
    p_corr = []
    if len(tr_counts) == 1:
        #Kolmogorov–Smirnov (KS) Test
        stat, ks_p_value = ks_2samp(tr_counts[0], tst_counts[0])
        stat = [stat]
        ks_p_value = [ks_p_value]
        # Jensen-Shannon Divergence (symmetric version of KL divergence)
        js_similarity = 1 - jensenshannon(tr_counts[0], tst_counts[0])  # Closer to 1 → more similar
        js_similarity = [js_similarity]
         # Pearson Correlation Coefficient
        p_corr, _ = pearsonr(tr_counts[0], tst_counts[0])
        p_corr = [p_corr]
    else:

        for i in range(len(tr_counts)):
            #Kolmogorov–Smirnov (KS) Test
            a, b = ks_2samp(tr_counts[i], tst_counts[i])
            stat.append(a)
            ks_p_value.append(b)
            # Jensen-Shannon Divergence (symmetric version of KL divergence)
            js_similarity.append(1 - jensenshannon(tr_counts[i], tst_counts[i]))  # Closer to 1 → more similar
            # Pearson Correlation Coefficient
            c, _ = pearsonr(tr_counts[i], tst_counts[i])
            p_corr.append(c)

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
    
    return ks_p_value, p_corr, js_similarity


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
        ks_p_value, p_corr, js_similarity = plot_train_test_target_distributions(train_loader, test_loader,data_name, num_outputs=num_outputs)
        if len(ks_p_value) == 1:

            print(f"KS Test p-value: {ks_p_value[0]}")
            print(f"Pearson Correlation Coefficient: {p_corr[0]}")
            print(f"Jensen-Shannon Divergence: {js_similarity[0]}")
        else:
            print(f"mean KS Test p-values: {np.mean(np.array(ks_p_value))}")
            print(f"mean Pearson Correlation Coefficients: {np.mean(np.array(p_corr))}")
            print(f"mean Jensen-Shannon Divergences: {np.mean(np.array(js_similarity))}")
            print(f"min KS Test p-values: {np.min(np.array(ks_p_value))}")
            print(f"min Pearson Correlation Coefficients: {np.min(np.array(p_corr))}")
            print(f"min Jensen-Shannon Divergences: {np.min(np.array(js_similarity))}")

    # if corr:
        # print("Calculating and plotting correlation matrix...")
        # # Assuming train_loader contains the training data
        # train_data = next(iter(train_loader))[0].cpu().numpy()  # Get a batch of data
        # corr_matrix = compute_feature_correlation(train_data, features)
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
        # plt.title("Correlation Matrix")
        # plt.show()
