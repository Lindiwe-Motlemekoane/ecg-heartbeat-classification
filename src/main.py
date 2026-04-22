#%%
# Step 1: Data Understanding

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

print("\n--- Loading Dataset ---")

df = pd.read_csv('../data/ecg_curated.csv', header=None)

print(df.head())
print("Shape:", df.shape)

# Separate features and label
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("\nClass distribution:")
print(y.value_counts())

print("\nFeature statistics:")
print(X.describe())

print("\nMissing values per column:")
print(df.isnull().sum())

# Check number of classes
num_classes = y.nunique()
print("\nNumber of classes:", num_classes)

if num_classes > 2:
    print("The label column is MULTICLASS")
else:
    print("The label column is BINARY")


#%%
# Plot sample heartbeats

print("\n--- Plotting Sample Heartbeats ---")

for i in range(3):
    plt.plot(X.iloc[i])
    plt.title(f"Heartbeat {i}")
    plt.xlabel("Time Step")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


#%%
# Normal vs Abnormal Comparison


print("\n--- Normal vs Abnormal Comparison ---")

# FIXED LABELS: 1 = normal, 0 = abnormal
normal_idx = df[df.iloc[:, -1] == 1].index[0]
abnormal_idx = df[df.iloc[:, -1] == 0].index[0]

normal = df.iloc[normal_idx, :-1].values
abnormal = df.iloc[abnormal_idx, :-1].values

x_axis = np.arange(len(normal))

plt.plot(x_axis, normal, label="Normal")
plt.plot(x_axis, abnormal, label="Abnormal")

plt.xticks(np.arange(0, len(normal), 20))
plt.xlabel("Time Step")
plt.ylabel("Amplitude")
plt.title("Normal vs Abnormal Heartbeat")
plt.legend()
plt.grid(True)
plt.show()


#%%
# Step 2: Feature Engineering (ABT)

from scipy.signal import find_peaks

print("\n--- Feature Engineering ---")

features = []
labels = []

# Iterating row-wise due to variable-length signals
for _, row in df.iterrows():
    values = row.dropna().values

    label = values[-1]
    signal = values[:-1]

    labels.append(label)

    # Statistical features
    mean = np.mean(signal)
    std = np.std(signal)
    max_R = np.max(signal)
    min_val = np.min(signal)
    range_val = max_R - min_val

    # Signal-based features
    energy = np.sum(signal ** 2)  # total signal intensity
    peaks, _ = find_peaks(signal)
    num_peaks = len(peaks)

    features.append([mean, std, max_R, min_val, range_val, energy, num_peaks])

# Create ABT
X_abt = pd.DataFrame(features, columns=['mean', 'std', 'max_R', 'min', 'range', 'energy', 'num_peaks'])
y_abt = pd.Series(labels, name='label')

abt = pd.concat([X_abt, y_abt], axis=1)

print(abt.head())


#%%
# Scaling for PCA

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_abt)


#%%
# Step 3: Correlation Analysis

print("\n--- Correlation Analysis ---")

corr_matrix = abt.corr()
print(corr_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


#%%
# Step 4: PCA

from sklearn.decomposition import PCA

print("\n--- PCA ---")

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])

print(pca_df.head())
print("\nVariance explained:", pca.explained_variance_ratio_)

total_variance = pca.explained_variance_ratio_.sum()
print("Total variance explained:", total_variance)


#%%
# Step 5: PCA Correlation

print("\n--- PCA Correlation ---")

pca_with_label = pd.concat([pca_df, y_abt], axis=1)
pca_corr = pca_with_label.corr()

print(pca_corr)

plt.figure(figsize=(8, 6))
sns.heatmap(pca_corr, annot=True, cmap='coolwarm')
plt.title("PCA Correlation Matrix")
plt.show()


#%%
# Step 6: Train/Test Split

print("\n--- Train/Test Split ---")

n = len(abt)
split_index = int((3/5) * n)

subset_A = abt.iloc[:split_index]
subset_B = abt.iloc[split_index:]

print("Subset A:", len(subset_A))
print("Subset B:", len(subset_B))

print("\nClass distribution (Full):")
print(abt['label'].value_counts(normalize=True))

print("\nSubset A:")
print(subset_A['label'].value_counts(normalize=True))

print("\nSubset B:")
print(subset_B['label'].value_counts(normalize=True))


#%%
# Step 7: Decision Tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, f1_score

print("\n--- Decision Tree ---")

X_A = subset_A.drop(columns=['label'])
y_A = subset_A['label']

X_B = subset_B.drop(columns=['label'])
y_B = subset_B['label']

dt = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

dt.fit(X_A, y_A)
y_pred_dt = dt.predict(X_B)

cm_dt = confusion_matrix(y_B, y_pred_dt)
print("Confusion Matrix (DT):\n", cm_dt)

sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues')
plt.title("Decision Tree Confusion Matrix")
plt.show()


#%%
# Step 8: Random Forest

from sklearn.ensemble import RandomForestClassifier

print("\n--- Random Forest ---")

rf = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42
)

rf.fit(X_A, y_A)
y_pred_rf = rf.predict(X_B)

cm_rf = confusion_matrix(y_B, y_pred_rf)
print("Confusion Matrix (RF):\n", cm_rf)

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
plt.title("Random Forest Confusion Matrix")
plt.show()


#%%
# Step 9: F1 Score Comparison

print("\n--- F1 Score Comparison ---")

f1_dt = f1_score(y_B, y_pred_dt)
f1_rf = f1_score(y_B, y_pred_rf)

print("Decision Tree F1:", f1_dt)
print("Random Forest F1:", f1_rf)


#%%
# Step 10: Cross Validation

from sklearn.model_selection import cross_val_score, cross_val_predict
from scipy.stats import ttest_rel

print("\n--- Cross Validation ---")

X_full = abt.drop(columns=['label'])
y_full = abt['label']

f1_dt_scores = cross_val_score(dt, X_full, y_full, cv=10, scoring='f1')
f1_rf_scores = cross_val_score(rf, X_full, y_full, cv=10, scoring='f1')

print("DT F1 scores:", f1_dt_scores)
print("RF F1 scores:", f1_rf_scores)

print("Average DT F1:", f1_dt_scores.mean())
print("Average RF F1:", f1_rf_scores.mean())

# Statistical significance test
t_stat, p_value = ttest_rel(f1_dt_scores, f1_rf_scores)
print("p-value:", p_value)


#%%
if __name__ == "__main__":
    print("\n--- ECG Classification Pipeline Completed ---")