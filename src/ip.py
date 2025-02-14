#!/usr/bin/env python3

import os
os.environ['MPLCONFIGDIR'] = '/tmp/'

import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for environments without a display

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import itertools
import random
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import time

from scipy.spatial.distance import cdist  # Import for computing distances

# ============================
# Step 1: Define Early Stopping Mechanism
# ============================

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

# ============================
# Step 2: Define GP_InducingPoints Class
# ============================

class GP_InducingPoints(torch.nn.Module):
    def __init__(self, _x=None, _y=None, _x_original=None, _num_inducing_points=None, x_m=None, x_std=None):
        super().__init__()

        # Assertions to ensure all necessary inputs are provided
        assert _x is not None, "Input data (_x) must be provided."
        assert _y is not None, "Output data (_y) must be provided."
        assert _x_original is not None, "Original input data (_x_original) must be provided."
        assert x_m is not None and x_std is not None, "Mean and std of X must be provided."
        assert _num_inducing_points >= 1, "Number of inducing points must be at least 1."

        self.x = _x  # Training input data: Tensor of shape [N, D]
        self.y = _y  # Training target data: Tensor of shape [N, 1]
        self.x_original = _x_original  # Original input data: NumPy array of shape [N, D]
        self.num_inducing_points = _num_inducing_points
        self.x_m = x_m  # Mean of X (for unscaling)
        self.x_std = x_std  # Std of X (for unscaling)

        # Compute min and max in scaled space for clamping
        self.x_min_scaled = self.x.min(dim=0).values  # Shape: [D]
        self.x_max_scaled = self.x.max(dim=0).values  # Shape: [D]

        # Perform K-Means clustering on the scaled input data to find inducing point locations
        kmeans = KMeans(n_clusters=self.num_inducing_points, n_init=10, random_state=random_seed)
        kmeans.fit(self.x.cpu().numpy())
        inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)

        # Clamp inducing points within the scaled data range
        inducing_points = torch.clamp(inducing_points, min=self.x_min_scaled, max=self.x_max_scaled)

        # Initialize inducing y-values based on nearest data points
        inducing_points_np = inducing_points.detach().cpu().numpy()
        inducing_points_unscaled = inducing_points_np * self.x_std + self.x_m  # Unscale
        inducing_points_untransformed = inducing_points_unscaled.copy()

        # Reverse log10 transformation for 'P' (assuming 'P' is the first feature)
        inducing_points_untransformed[:, 0] = 10 ** inducing_points_unscaled[:, 0]

        # Compute distances between inducing points and original data points
        distances = cdist(inducing_points_untransformed, self.x_original)  # Shape: [num_inducing_points, N]
        nearest_indices = np.argmin(distances, axis=1)  # Shape: [num_inducing_points]

        # Assign y-values based on the nearest data points (scaled y)
        inducing_ys = self.y[nearest_indices]  # Shape: [num_inducing_points, 1]

        # Initialize inducing point parameters
        self.inducing_x_mu = torch.nn.Parameter(inducing_points.clone().detach())
        self.inducing_y_mu = torch.nn.Parameter(inducing_ys.clone().detach())

        # Store initial inducing points for potential resetting
        self.inducing_x_mu_init = self.inducing_x_mu.clone().detach()
        self.inducing_y_mu_init = self.inducing_y_mu.clone().detach()

        # Initialize kernel hyperparameters
        self.length_scale = torch.nn.Parameter(torch.tensor(1.0, device=device))
        self.noise = torch.nn.Parameter(torch.tensor(0.1, device=device))

    def reset_inducing_points(self):
        # Reset inducing points to their initial values
        with torch.no_grad():
            self.inducing_x_mu.copy_(self.inducing_x_mu_init)
            self.inducing_y_mu.copy_(self.inducing_y_mu_init)

    def compute_kernel_matrix(self, x1, x2):
        """
        Compute the Rational Quadratic Kernel matrix between x1 and x2.

        Args:
            x1: Tensor of shape [N, D]
            x2: Tensor of shape [M, D]

        Returns:
            Kernel matrix of shape [N, M]
        """
        # Compute the pairwise squared Euclidean distances
        pdist = torch.cdist(x1, x2, p=2) ** 2  # Shape: [N, M]
        kernel_matrix = (1 + pdist / (2 * self.length_scale ** 2)) ** (-1)  # Rational Quadratic Kernel
        return kernel_matrix

    def compute_kernel_diag(self, x):
        """
        Compute the diagonal elements of the kernel matrix for x.

        Args:
            x: Tensor of shape [N, D]

        Returns:
            Diagonal of the kernel matrix: Tensor of shape [N]
        """
        return torch.ones(x.shape[0], device=device)

    def forward(self, _X):
        K_XsX = self.compute_kernel_matrix(_X, self.inducing_x_mu)  # Shape: [N, M]
        K_XX = self.compute_kernel_matrix(self.inducing_x_mu, self.inducing_x_mu)  # Shape: [M, M]
        K_XX += 1e-6 * torch.eye(K_XX.shape[0], device=device)  # Add jitter for stability
        K_XX_inv = torch.inverse(K_XX)

        mu = K_XsX @ K_XX_inv @ self.inducing_y_mu  # Shape: [N, 1]

        # Compute predictive variance
        K_XsXs_diag = self.compute_kernel_diag(_X)  # Shape: [N]
        Q_XsXs_diag = (K_XsX @ K_XX_inv * K_XsX).sum(dim=1)  # Shape: [N]

        var = K_XsXs_diag - Q_XsXs_diag + self.noise ** 2  # Shape: [N]
        var = var.clamp_min(1e-6)  # Ensure variance is positive

        return mu, var.unsqueeze(1)  # Shapes: [N, 1], [N, 1]

    def NLL(self, _X, _y):
        """
        Compute the Negative Log Likelihood for GP regression.

        Args:
            _X: Input data, Tensor of shape [N, D]
            _y: Target data, Tensor of shape [N, 1]

        Returns:
            Negative Log Likelihood scalar
        """
        K_XsX = self.compute_kernel_matrix(_X, self.inducing_x_mu)  # Shape: [N, M]
        K_XX = self.compute_kernel_matrix(self.inducing_x_mu, self.inducing_x_mu)  # Shape: [M, M]
        K_XX += 1e-6 * torch.eye(K_XX.shape[0], device=device)  # Add jitter for stability
        K_XX_inv = torch.inverse(K_XX)

        mu = K_XsX @ K_XX_inv @ self.inducing_y_mu  # Shape: [N, 1]

        # Compute predictive variance
        K_XsXs_diag = self.compute_kernel_diag(_X)  # Shape: [N]
        Q_XsXs_diag = (K_XsX @ K_XX_inv * K_XsX).sum(dim=1)  # Shape: [N]

        var = K_XsXs_diag - Q_XsXs_diag + self.noise ** 2  # Shape: [N]
        var = var.clamp_min(1e-6)  # Ensure variance is positive

        # Compute Negative Log Likelihood
        residual = (_y.squeeze() - mu.squeeze())  # Shape: [N]
        nll = 0.5 * torch.log(2 * np.pi * var) + 0.5 * (residual ** 2) / var  # Shape: [N]
        return nll.mean()

    def compute_lambda(self):
        """
        Compute the lambda values for the training data.

        Returns:
            lambda_i: Tensor of shape [N]
        """
        K_XsX = self.compute_kernel_matrix(self.x, self.inducing_x_mu)  # Shape: [N, M]
        K_XX = self.compute_kernel_matrix(self.inducing_x_mu, self.inducing_x_mu)  # Shape: [M, M]
        K_XX += 1e-6 * torch.eye(K_XX.shape[0], device=device)  # Add jitter for stability
        K_XX_inv = torch.inverse(K_XX)

        # Compute Q_XsXs_diag
        Q_XsXs_diag = (K_XsX @ K_XX_inv * K_XsX).sum(dim=1)  # Shape: [N]

        # Compute lambda_i
        K_XsXs_diag = self.compute_kernel_diag(self.x)  # Shape: [N]
        lambda_i = K_XsXs_diag - Q_XsXs_diag  # Shape: [N]
        lambda_i = lambda_i.clamp_min(1e-6)  # Ensure lambda_i is positive

        return lambda_i

# ============================
# Step 3: Set Up Environment and Reproducibility
# ============================

# Start timer
start_time = time.time()

# Set random seeds for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# Determine device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define FloatTensor based on device
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# ============================
# Step 4: Load and Preprocess Data
# ============================

# Load CSV data
data = pd.read_csv("Main.csv")

# Ensure required columns are present and numeric
required_columns = ['P', 'VF', 'LCD', 'PLD', 'SA', 'y']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"Column '{col}' not found in the dataset.")
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with NaN values in required columns
initial_row_count = data.shape[0]
data.dropna(subset=required_columns, inplace=True)
if data.shape[0] < initial_row_count:
    print(f"Warning: Dropped {initial_row_count - data.shape[0]} rows due to NaN values.")

# Remove non-positive values for 'P' and 'y' before log10 transformation
data = data[(data['P'] > 0) & (data['y'] > 0)]
print(f"Data shape after filtering non-positive 'P' and 'y': {data.shape}")

# Extract features and targets
X_original = data[['P', 'VF', 'LCD', 'PLD', 'SA']].values  # Shape: [N, 5]
y_original = data['y'].values.reshape(-1, 1)               # Shape: [N, 1]

# Apply log10 transformation to 'P' and 'y'
X_transformed = X_original.copy()
X_transformed[:, 0] = np.log10(X_transformed[:, 0])      # Log-transform 'P'
y_transformed = np.log10(y_original)                     # Log-transform 'y'

# Verify no infinities or NaNs after log transformation
if np.isnan(X_transformed).any() or np.isinf(X_transformed).any():
    raise ValueError("NaN or infinity detected in X_transformed after log10 transformation.")
if np.isnan(y_transformed).any() or np.isinf(y_transformed).any():
    raise ValueError("NaN or infinity detected in y_transformed after log10 transformation.")

# Compute mean and standard deviation for scaling
x_m = X_transformed.mean(axis=0)
x_std = X_transformed.std(axis=0)
y_m = y_transformed.mean()
y_std = y_transformed.std()

# Prevent division by zero in scaling
x_std[x_std == 0] = 1e-6
if y_std == 0:
    y_std = 1e-6

# Scale the data
X_scaled = (X_transformed - x_m) / x_std
y_scaled = (y_transformed - y_m) / y_std

# Split into training and validation sets (80-20 split)
X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled, X_train_original, X_val_original, y_train_original, y_val_original = train_test_split(
    X_scaled, y_scaled, X_original, y_original, test_size=0.2, random_state=random_seed
)

# Convert to FloatTensor
X_train = FloatTensor(X_train_scaled)
y_train = FloatTensor(y_train_scaled)
X_val = FloatTensor(X_val_scaled)
y_val = FloatTensor(y_val_scaled)

# Debugging: Print statistics of scaled data
print(f"Scaled Training X statistics: mean={X_train.mean().item():.4f}, std={X_train.std().item():.4f}")
print(f"Scaled Training y statistics: mean={y_train.mean().item():.4f}, std={y_train.std().item():.4f}")
print(f"Scaled Validation X statistics: mean={X_val.mean().item():.4f}, std={X_val.std().item():.4f}")
print(f"Scaled Validation y statistics: mean={y_val.mean().item():.4f}, std={y_val.std().item():.4f}")

# ============================
# Step 5: Define Hyperparameters
# ============================

# Define the hyperparameters to search over
# Adjust the hyperparameter grid based on computational resources
num_samples = [500]           # Batch sizes
num_inducing_points = [500, 1000, 2000]      # Number of inducing points
lr_values = [0.001, 0.01]             # Learning rates
num_epochs = [100, 200]                # Number of epochs

# Create all combinations of hyperparameters for grid search
configs = list(itertools.product(num_samples, num_inducing_points, lr_values, num_epochs))
print(f"Total number of hyperparameter configurations: {len(configs)}")

# ============================
# Step 6: Compute Lambda Values for Each Inducing Point Configuration
# ============================

# Initialize DataFrame to store lambda values
lambda_values_df = pd.DataFrame(index=range(len(X_train_original)))

for num_i in num_inducing_points:
    print(f"Computing lambda for num_inducing_points={num_i}")
    gp_model = GP_InducingPoints(
        _x=X_train,
        _y=y_train,
        _x_original=X_train_original,
        _num_inducing_points=num_i,
        x_m=x_m,
        x_std=x_std
    ).to(device)
    lambda_values = gp_model.compute_lambda().detach().cpu().numpy()
    lambda_values_df[num_i] = lambda_values

# Save lambda values to a CSV file
lambda_values_df.to_csv("lambda_values_by_inducing_points.csv", index=False)
print("Saved lambda values to 'lambda_values_by_inducing_points.csv'.")

# ============================
# Step 7: Define Training and Evaluation Function
# ============================

def train_and_evaluate(config):
    num_s, num_i, lr_k, epochs = config
    print(f"\nStarting training for config: num_samples={num_s}, num_inducing_points={num_i}, lr={lr_k}, num_epochs={epochs}")

    # Initialize the GP model
    gp = GP_InducingPoints(
        _x=X_train,
        _y=y_train,
        _x_original=X_train_original,
        _num_inducing_points=num_i,
        x_m=x_m,
        x_std=x_std
    ).to(device)

    # Create DataLoaders for training and validation
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=num_s, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=num_s, shuffle=False)

    # Initialize optimizer
    optim = torch.optim.Adam(gp.parameters(), lr=lr_k)

    # Initialize early stopping with adjusted min_delta
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)

    # Training loop with early stopping
    gp.train()
    for epoch in range(epochs):
        epoch_losses = []
        for data_batch, label_batch in train_loader:
            optim.zero_grad()
            nll = gp.NLL(data_batch, label_batch)
            nll.backward()
            optim.step()

            # Clamp inducing_x_mu to ensure they stay within the data range
            with torch.no_grad():
                gp.inducing_x_mu.clamp_(min=gp.x_min_scaled, max=gp.x_max_scaled)

            epoch_losses.append(nll.item())

        # Compute average training loss for the epoch
        avg_train_loss = np.mean(epoch_losses)

        # Compute validation loss
        gp.eval()
        with torch.no_grad():
            val_losses = []
            for val_batch, val_label in val_loader:
                val_nll = gp.NLL(val_batch, val_label)
                val_losses.append(val_nll.item())
            avg_val_loss = np.mean(val_losses)
        gp.train()

        print(f"Epoch {epoch+1}/{epochs}, Train NLL: {avg_train_loss:.6f}, Val NLL: {avg_val_loss:.6f}")

        # Check early stopping condition
        if early_stopping(avg_val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Evaluation on validation set
    gp.eval()
    with torch.no_grad():
        mu, var = gp.forward(X_val)  # mu: [N_val, 1], var: [N_val, 1]
        mu = mu.cpu().numpy() * y_std + y_m  # Unscale
        y_val_true = y_val.cpu().numpy() * y_std + y_m  # Unscale

    # Debugging: Check ranges
    print(f"After training - Validation y_true: min={y_val_true.min():.4f}, max={y_val_true.max():.4f}")
    print(f"After training - Validation mu: min={mu.min():.4f}, max={mu.max():.4f}")

    # Compute MAE and R² in normal space with safe exponentiation
    # Cap the exponentials to prevent overflow
    max_exponent = 38  # Maximum exponent before float32 overflow (10**38 ~ 3.4e38)
    y_val_exp = np.power(10, np.clip(y_val_true, a_min=None, a_max=max_exponent))
    mu_exp = np.power(10, np.clip(mu, a_min=None, a_max=max_exponent))

    # Compute MAE and R² in original space
    mae_norm = mean_absolute_error(y_val_exp, mu_exp)
    r2_norm = r2_score(y_val_exp, mu_exp)
    print(f"Validation R2 (normal space): {r2_norm:.4f}, Validation MAE (normal space): {mae_norm:.4f}")

    return {
        'mae_norm': mae_norm,
        'r2_norm': r2_norm,
        'config': config,
        'state_dict': gp.state_dict(),
        'length_scale': gp.length_scale.item(),
        'noise': gp.noise.item()
    }

# ============================
# Step 8: Perform Grid Search Serially
# ============================

print("\nStarting grid search...")

results = []
for idx, config in enumerate(configs, start=1):
    print(f"\nConfiguration {idx}/{len(configs)}:")
    result = train_and_evaluate(config)
    results.append(result)

print("Grid search completed.")

# ============================
# Step 9: Identify the Best Configuration
# ============================

# Initialize variables to store the best results
best_mae_norm = float('inf')
best_config = None
best_model_state_dict = None
best_length_scale = None
best_noise = None
best_r2_norm = None

for result in results:
    mae_norm = result['mae_norm']
    if mae_norm < best_mae_norm:
        best_mae_norm = mae_norm
        best_config = result['config']
        best_model_state_dict = result['state_dict']
        best_length_scale = result['length_scale']
        best_noise = result['noise']
        best_r2_norm = result['r2_norm']

if best_model_state_dict is not None:
    print("\nBest Hyperparameters:")
    print(f"  num_samples={best_config[0]}")
    print(f"  num_inducing_points={best_config[1]}")
    print(f"  lr={best_config[2]}")
    print(f"  num_epochs={best_config[3]}")
    print(f"Best Validation MAE (normal space): {best_mae_norm:.4f}")
    print(f"Best Validation R2 (normal space): {best_r2_norm:.4f}")
    print(f"Optimized Length Scale: {best_length_scale:.4f}")
    print(f"Optimized Noise Level: {best_noise:.4f}")
else:
    print("No valid model found.")
    exit()

# ============================
# Step 10: Retrain the Best Model on the Entire Dataset
# ============================

print("\nRetraining the best model on the entire dataset...")

# Combine training and validation data
X_full_scaled = torch.cat((X_train, X_val), dim=0)
y_full_scaled = torch.cat((y_train, y_val), dim=0)
X_full_original = np.vstack((X_train_original, X_val_original))
y_full_original = np.vstack((y_train_original, y_val_original))

# Convert to FloatTensor
X_full = X_full_scaled
y_full = y_full_scaled

# Initialize the best model with the best hyperparameters
best_num_samples, best_num_inducing_points, best_lr, best_num_epochs = best_config

final_gp = GP_InducingPoints(
    _x=X_full,
    _y=y_full,
    _x_original=X_full_original,
    _num_inducing_points=best_num_inducing_points,
    x_m=x_m,
    x_std=x_std
).to(device)

# Create DataLoader for the entire dataset
final_dataset = TensorDataset(X_full, y_full)
final_loader = DataLoader(final_dataset, batch_size=best_num_samples, shuffle=True)

# Initialize optimizer with best learning rate
final_optim = torch.optim.Adam(final_gp.parameters(), lr=best_lr)

# Initialize Early Stopping with adjusted min_delta
final_early_stopping = EarlyStopping(patience=10, min_delta=1e-4)

# Training loop with early stopping
final_gp.train()
for epoch in range(best_num_epochs):
    epoch_losses = []
    for data_batch, label_batch in final_loader:
        final_optim.zero_grad()
        nll = final_gp.NLL(data_batch, label_batch)
        nll.backward()
        final_optim.step()

        # Clamp inducing_x_mu to ensure they stay within the data range
        with torch.no_grad():
            final_gp.inducing_x_mu.clamp_(min=final_gp.x_min_scaled, max=final_gp.x_max_scaled)

        epoch_losses.append(nll.item())

    # Compute average training loss for the epoch
    avg_train_loss = np.mean(epoch_losses)

    # Since we're training on the entire dataset, validation loss is same as training loss
    avg_val_loss = avg_train_loss

    print(f"Epoch {epoch+1}/{best_num_epochs}, Train NLL: {avg_train_loss:.6f}, Val NLL: {avg_val_loss:.6f}")

    # Check early stopping condition
    if final_early_stopping(avg_val_loss):
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

# ============================
# Step 11: Evaluate the Final Model on All Data
# ============================

final_gp.eval()
with torch.no_grad():
    mu_all, var_all = final_gp.forward(X_full)  # mu: [N_full, 1], var: [N_full, 1]
    mu_all = mu_all.cpu().numpy() * y_std + y_m  # Unscale
    sigma_all = np.sqrt(var_all.cpu().numpy()) * y_std  # Unscale
    y_full_true = y_full.cpu().numpy() * y_std + y_m  # Unscale

# Debugging: Check ranges
print(f"\nAfter final training - All Data y_true: min={y_full_true.min():.4f}, max={y_full_true.max():.4f}")
print(f"After final training - All Data mu: min={mu_all.min():.4f}, max={mu_all.max():.4f}")

# Compute MAE and R² in normal space with safe exponentiation
max_exponent = 38  # Maximum exponent before float32 overflow (10**38 ~ 3.4e38)
y_full_exp = np.power(10, np.clip(y_full_true, a_min=None, a_max=max_exponent))
mu_all_exp = np.power(10, np.clip(mu_all, a_min=None, a_max=max_exponent))

# Compute MAE and R² in original space
final_mae_norm = mean_absolute_error(y_full_exp, mu_all_exp)
final_r2_norm = r2_score(y_full_exp, mu_all_exp)

print(f"Final R2 (normal space): {final_r2_norm:.4f}, Final MAE (normal space): {final_mae_norm:.4f}")

# ============================
# Step 12: Save Predictions to CSV
# ============================

# Create DataFrame for predictions
results_df = pd.DataFrame({
    'P': 10 ** (X_full_scaled[:, 0].cpu().numpy()),  # Convert back to original 'P'
    'VF': X_full_original[:, 1],
    'LCD': X_full_original[:, 2],
    'PLD': X_full_original[:, 3],
    'SA': X_full_original[:, 4],
    'True_y': y_full_true.squeeze(),
    'Predicted_y': mu_all.squeeze(),
    'Sigma': sigma_all.squeeze()
})

# Include exponentiated values with capping to prevent overflow
results_df['True_y'] = np.power(10, np.clip(y_full_true.squeeze(), a_min=None, a_max=max_exponent))
results_df['Predicted_y'] = np.power(10, np.clip(mu_all.squeeze(), a_min=None, a_max=max_exponent))
results_df['Sigma'] = np.power(10, np.clip(sigma_all.squeeze(), a_min=None, a_max=max_exponent))

# Save predictions to CSV
results_df.to_csv('GP_predictions.csv', index=False)
print("Saved predictions to 'GP_predictions.csv'.")

# ============================
# Step 13: Save Inducing Points with Nearest Data Points
# ============================

# Unscale inducing points
inducing_points = final_gp.inducing_x_mu.detach().cpu().numpy()
inducing_points_unscaled = inducing_points * x_std + x_m  # Unscale

# Reverse log10 transformation for 'P'
inducing_points_untransformed = inducing_points_unscaled.copy()
inducing_points_untransformed[:, 0] = 10 ** inducing_points_unscaled[:, 0]  # Reverse log10 of 'P'

# Reverse scaling for y
inducing_y = final_gp.inducing_y_mu.detach().cpu().numpy() * y_std + y_m
inducing_y = 10 ** inducing_y  # Reverse log10 of 'y'

# Create DataFrame for inducing points
inducing_points_df = pd.DataFrame(inducing_points_untransformed, columns=['P', 'VF', 'LCD', 'PLD', 'SA'])
inducing_points_df['y'] = inducing_y.squeeze()

# Compute distances between final inducing points and all data points
distances = cdist(inducing_points_untransformed, X_full_original)  # Shape: [num_inducing_points, N_full]
nearest_indices = np.argmin(distances, axis=1)  # Shape: [num_inducing_points]

# Get nearest main data points
nearest_data_points = X_full_original[nearest_indices]  # Shape: [num_inducing_points, 5]
nearest_data_y = y_full_original[nearest_indices]      # Shape: [num_inducing_points, 1]

# Add nearest main data points to the DataFrame
inducing_points_df['Nearest_P'] = nearest_data_points[:, 0]
inducing_points_df['Nearest_VF'] = nearest_data_points[:, 1]
inducing_points_df['Nearest_LCD'] = nearest_data_points[:, 2]
inducing_points_df['Nearest_PLD'] = nearest_data_points[:, 3]
inducing_points_df['Nearest_SA'] = nearest_data_points[:, 4]
inducing_points_df['Nearest_y'] = nearest_data_y.squeeze()

# Save inducing points with nearest data points to CSV
inducing_points_df.to_csv("inducing_points_with_nearest_data.csv", index=False)
print("Saved inducing points with nearest data points to 'inducing_points_with_nearest_data.csv'.")

# ============================
# Step 14: Final Metrics and Timing
# ============================

print(f"\nFinal R2 (normal space): {final_r2_norm:.4f}, Final MAE (normal space): {final_mae_norm:.4f}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time in seconds: {elapsed_time:.2f}")
