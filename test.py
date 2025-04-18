import numpy as np
from scipy.stats import spearmanr, pearsonr

# Load your .npz file
data = np.load('margins_eval_dpo-margin-0-8b-2.npz')

# Print available keys to see what's inside
print("Available keys:", data.files)

# Extract the relevant arrays
# Replace 'array1' and 'array2' with actual keys from the file
arr1 = data['gt']
arr2 = data['pref']

pearson_corr, pearson_p = pearsonr(arr1, arr2)
print(f"Pearson Correlation: {pearson_corr:.4f}, p-value: {pearson_p:.4e}")

# Compute Spearman's rank correlation coefficient
correlation, p_value = spearmanr(arr1, arr2)

print(f"Spearman's correlation: {correlation:.4f}")
print(f"P-value: {p_value:.4e}")
print(f"Acc: {(arr2 >= 0).mean()}")
