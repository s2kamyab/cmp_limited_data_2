import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
def train_test_split_time_series(df, test_size=0.1):
    """
    Splits a time series dataframe into train and test sets by time order.
    """
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df
# === CONFIG ===
csv_path = ".\\data_clarckson\\20250708_VLCC_after_MCV.csv"    # Replace with your path
target_column = "42930"            # Replace with your actual target column name
test_size = 0.2                     # 20% test split
n_components = 8                    # Number of PCA components
output_train_csv = rf'data_clarckson\\{target_column}_train_pca.csv'
output_test_csv = rf'data_clarckson\\{target_column}_test_pca.csv'
df = pd.read_csv(r'data_clarckson\\20250708_VLCC_after_MCV.csv')
all_targets = ['42930','95900', '47353', '541976']
# df['time_step'] = range(len(df))
# # df = df.drop('date', axis=1)
# columns_to_normalize = range(len(df.columns))
# columns_to_normalize = [columns_to_normalize[:-2]] # exclude sentiment column
# if use_sentiment == 0:
#     df = df.drop('Sentiment_textblob', axis=1)
# else:
#     df['Sentiment_textblob'] = df['Sentiment_textblob'].shift(use_sentiment).fillna(0)

# df = df.drop('95900', axis=1)
# df = df.drop('47353', axis=1)
# df = df.drop('541976', axis=1)

# train1, test1 = train_test_split_time_series(df, test_size=0.3)

df = pd.read_csv(csv_path)
assert target_column in df.columns, f"Target column '{target_column}' not found in data."
X = df.copy()
for i in all_targets:
    # if i != target_column:
    X = X.drop(i, axis=1)
# === STEP 2: Split into X (features) and y (target) ===
X = X.drop(columns=['Sentiment_textblob'])#, 'date'])
y = df[target_column]
# target_index = df.columns.to_list().index(target_column) 

# === STEP 3: Split into train and test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, shuffle=False  # Set shuffle=True for non-time series
)

# === STEP 4: Standardize and apply PCA ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# === STEP 5: Combine PCA-transformed features with unmodified targets ===
train_pca_df = pd.DataFrame(X_train_pca, columns=[f'pca_{i+1}' for i in range(n_components)])
train_pca_df[target_column] = y_train.reset_index(drop=True)
train_pca_df['Sentiment_textblob'] = df['Sentiment_textblob'].iloc[:len(train_pca_df)].reset_index(drop=True)
# train_pca_df['date'] = df['date'].iloc[:len(train_pca_df)].reset_index(drop=True)

test_pca_df = pd.DataFrame(X_test_pca, columns=[f'pca_{i+1}' for i in range(n_components)])
test_pca_df[target_column] = y_test.reset_index(drop=True)
test_pca_df['Sentiment_textblob'] = df['Sentiment_textblob'].iloc[len(train_pca_df):].reset_index(drop=True)

# === STEP 6: Save to CSV ===
train_pca_df.to_csv(output_train_csv, index=False)
test_pca_df.to_csv(output_test_csv, index=False)

print("✅ Saved:")
print(f"  • {output_train_csv}")
print(f"  • {output_test_csv}")
