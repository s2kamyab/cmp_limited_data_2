import matplotlib.pyplot as plt
import seaborn as sns
import torch
def plot_train_test_target_distributions(train_loader, test_loader, num_outputs=3):
    # Step 1: Extract targets from DataLoaders
    def extract_targets(loader):
        all_targets = []
        for _, y in loader:
            all_targets.append(y[0])
        return torch.cat(all_targets, dim=0).cpu().numpy()  # shape: (N, num_outputs)

    train_targets = extract_targets(train_loader)
    test_targets = extract_targets(test_loader)

    # Step 2: Plot for each output dimension
    for i in range(num_outputs):
        plt.figure(figsize=(6, 4))
        sns.kdeplot(train_targets[:, i], label='Train', fill=True)
        sns.kdeplot(test_targets[:, i], label='Test', fill=True)
        plt.title(f"Distribution of Target {i + 1}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
