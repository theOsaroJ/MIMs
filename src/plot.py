#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1) Load the data
# -----------------------------
df = pd.read_csv("compare.csv")

# Actual (first column) and predicted (second column) using iloc
y_true = df.iloc[:, 0].values
y_pred = df.iloc[:, 1].values

# -----------------------------
# 2) Compute metrics
# -----------------------------
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
r2  = r2_score(y_true, y_pred)

print(f"R²:  {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")

# -----------------------------
# 3) Parity Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter
ax.scatter(y_true, y_pred, alpha=0.7, edgecolors='k', label='Data points')

# y=x line
min_val = min(np.min(y_true), np.min(y_pred))
max_val = max(np.max(y_true), np.max(y_pred))
ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x line')

# Customize
ax.set_xlabel(r'Actual Adsorption $[cm^3 (stp)/ gr\:framework]$', fontsize=20)
ax.set_ylabel(r'Predicted Adsorption $[cm^3 (stp)/ gr\:framework]$', fontsize=20)
# ax.set_title("Parity Plot", fontsize=16, fontweight='bold')  # optional
ax.legend(fontsize=20)

# Adjust tick label sizes
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)

# Layout & save
plt.tight_layout()
plt.savefig("parity_plot.png", dpi=300)
plt.close()
print("Saved 'parity_plot.png'")

# -----------------------------
# 4) Error Box Plot
# -----------------------------
errors = y_true - y_pred

fig2, ax2 = plt.subplots(figsize=(12, 8))

# Boxplot
boxprops = dict(facecolor='lightblue', alpha=0.6)  # example style
ax2.boxplot(errors, vert=True, patch_artist=True, boxprops=boxprops)

# Customize
ax2.set_ylabel('Error (Actual - Predicted)', fontsize=25)
# ax2.set_title('Error Distribution', fontsize=16, fontweight='bold')  # optional

ax2.tick_params(axis='x', labelsize=25)
ax2.tick_params(axis='y', labelsize=25)

plt.tight_layout()
plt.savefig("error_boxplot.png", dpi=300)
plt.close()
print("Saved 'error_boxplot.png'")
