#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.stats import norm

########################################
# GaussianProcess Class with Combined Kernel
########################################
class GaussianProcess:
    def __init__(self, length_scale=1.0, noise=0.1, batch_size=2000):
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.batch_size = batch_size
        self.alpha_rq = 1.0  # parameter for RQ kernel

    def rbf_kernel(self, XA, XB, length_scale=None):
        if length_scale is None:
            length_scale = self.length_scale
        sqdist = cdist(XA, XB, 'sqeuclidean')
        K = np.exp(-0.5 * sqdist / (length_scale**2))
        return K

    def rq_kernel(self, XA, XB, length_scale=None, alpha=None):
        if length_scale is None:
            length_scale = self.length_scale
        if alpha is None:
            alpha = self.alpha_rq
        sqdist = cdist(XA, XB, 'sqeuclidean')
        factor = 1 + sqdist/(2*alpha*(length_scale**2))
        K = factor**(-alpha)
        return K

    def matern_32_kernel(self, XA, XB, length_scale=None):
        if length_scale is None:
            length_scale = self.length_scale
        r = cdist(XA, XB, 'euclidean')
        sqrt3 = np.sqrt(3)
        scaled_r = sqrt3 * r / length_scale
        K = (1.0 + scaled_r)*np.exp(-scaled_r)
        return K

    def combined_kernel(self, XA, XB, length_scale=None, noise=None, include_noise=False):
        if length_scale is None:
            length_scale = self.length_scale
        if noise is None:
            noise = self.noise

        K_rbf = self.rbf_kernel(XA, XB, length_scale=length_scale)
        K_rq = self.rq_kernel(XA, XB, length_scale=length_scale, alpha=self.alpha_rq)
        K_matern = self.matern_32_kernel(XA, XB, length_scale=length_scale)

        # Combine equally
        K = (K_rbf + K_rq + K_matern) / 3.0

        if include_noise and XA.shape[0] == XB.shape[0]:
            K += (noise**2)*np.eye(XA.shape[0])
        return K

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        K = self.combined_kernel(X_train, X_train, include_noise=True)
        self.K_inv = np.linalg.inv(K)

    def predict_in_batches(self, X_test):
        N_test = X_test.shape[0]
        mu = np.zeros(N_test)
        var = np.zeros(N_test)
        for start in range(0, N_test, self.batch_size):
            end = min(start+self.batch_size, N_test)
            X_batch = X_test[start:end]

            K_star = self.combined_kernel(X_batch, self.X_train)
            mu_batch = K_star @ self.K_inv @ self.y_train

            K_starstar = self.combined_kernel(X_batch, X_batch)
            cov_batch = K_starstar - K_star @ self.K_inv @ K_star.T
            var_batch = np.diag(cov_batch)

            mu[start:end] = mu_batch.ravel()
            var[start:end] = var_batch.ravel()
        return mu, var

    def predict(self, X_test):
        return self.predict_in_batches(X_test)

    def neg_log_marginal_likelihood(self, params):
        length_scale = np.exp(params[0])
        noise = np.exp(params[1])

        K = self.combined_kernel(self.X_train, self.X_train,
                                 length_scale=length_scale,
                                 noise=noise,
                                 include_noise=True)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return np.inf  # numerical issue

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
        n = self.y_train.shape[0]
        ll = -0.5 * self.y_train.T @ alpha
        ll -= np.sum(np.log(np.diag(L)))
        ll -= 0.5 * n * np.log(2*np.pi)
        return -ll.ravel()[0]

########################################
# Acquisition Functions
########################################

def acquisition_ucb(mu, var, beta=1.0):
    return mu + beta*np.sqrt(var)

def acquisition_std(mu, var):
    return np.sqrt(var)

def acquisition_ei(mu, var, best_y, xi=0.01):
    sigma = np.sqrt(var)
    improvement = mu - best_y - xi
    Z = np.zeros_like(mu)
    # Only compute Z where sigma>0 to avoid division by zero
    nonzero = sigma>0
    Z[nonzero] = improvement[nonzero]/sigma[nonzero]
    ei = np.zeros_like(mu)
    ei[nonzero] = sigma[nonzero]*(Z[nonzero]*norm.cdf(Z[nonzero]) + norm.pdf(Z[nonzero]))
    return ei

def acquisition_pi(mu, var, best_y, xi=0.01):
    sigma = np.sqrt(var)
    improvement = mu - best_y - xi
    Z = np.zeros_like(mu)
    nonzero = sigma>0
    Z[nonzero] = improvement[nonzero]/sigma[nonzero]
    pi = np.zeros_like(mu)
    pi[nonzero] = norm.cdf(Z[nonzero])
    return pi

# We'll dynamically choose based on user input
# Some acquisitions need best_y, which we must get after GP fit each iteration.
ACQUISITION_FUNCS = {
    'ucb': acquisition_ucb,
    'std': acquisition_std,
    'ei': acquisition_ei,
    'pi': acquisition_pi
}

########################################
# Active Learning with query_size
########################################
def active_learning_run(X_scaled, y_scaled, num_points, acquisition='ucb', length_scale=1.0, noise=0.1, random_seed=42, query_size=1):
    """
    Active learning loop that selects 'query_size' points each iteration until num_points reached.
    """
    np.random.seed(random_seed)
    N = X_scaled.shape[0]

    # Initial selection: K-Means + random
    initial_k = 2
    kmeans = KMeans(n_clusters=initial_k, n_init=10, random_state=random_seed)
    kmeans.fit(X_scaled)
    centers = kmeans.cluster_centers_
    dist_centers = cdist(centers, X_scaled)
    nearest_idx = np.argmin(dist_centers, axis=1)
    selected_indices = list(nearest_idx)

    num_random = 2
    all_idx = set(range(N))
    remaining = list(all_idx - set(selected_indices))
    random_sel = np.random.choice(remaining, num_random, replace=False)
    selected_indices.extend(random_sel)

    unlabeled_indices = list(all_idx - set(selected_indices))

    gp = GaussianProcess(length_scale=length_scale, noise=noise, batch_size=2000)
    gp.fit(X_scaled[selected_indices], y_scaled[selected_indices])

    if acquisition not in ACQUISITION_FUNCS:
        raise ValueError(f"Acquisition '{acquisition}' not implemented. Choose from {list(ACQUISITION_FUNCS.keys())}")
    acq_func = ACQUISITION_FUNCS[acquisition]

    xi = 0.01
    while len(selected_indices) < num_points:
        mu_all, var_all = gp.predict(X_scaled)

        # Compute best_y from current training points (scaled space)
        current_best_y = np.max(gp.y_train)

        # Depending on acquisition, call with correct parameters
        if acquisition in ['ei', 'pi']:
            scores = acq_func(mu_all, var_all, current_best_y, xi=xi)
        elif acquisition == 'ucb':
            scores = acq_func(mu_all, var_all, beta=1.0)
        else:
            # std or others that don't need additional params
            scores = acq_func(mu_all, var_all)

        # focus on unlabeled
        unlabeled_scores = scores[unlabeled_indices]

        points_to_add = min(query_size, num_points - len(selected_indices))
        top_indices = np.argpartition(unlabeled_scores, -points_to_add)[-points_to_add:]
        top_indices = top_indices[np.argsort(unlabeled_scores[top_indices])][::-1]

        chosen_global_indices = [unlabeled_indices[i] for i in top_indices]

        selected_indices.extend(chosen_global_indices)
        for cidx in chosen_global_indices:
            unlabeled_indices.remove(cidx)

        gp.fit(X_scaled[selected_indices], y_scaled[selected_indices])

        if len(selected_indices) >= num_points:
            break

    # Compute mean uncertainty
    _, var_final = gp.predict(X_scaled)
    mean_var = np.mean(var_final)
    return mean_var, selected_indices, gp

def optimize_hyperparameters(gp):
    """
    Optimize length_scale and noise by minimizing NMLL.
    """
    init_params = [np.log(gp.length_scale), np.log(gp.noise)]

    def obj(params):
        return gp.neg_log_marginal_likelihood(params)

    res = minimize(obj, init_params, bounds=[(-5,5),(-5,5)], method='L-BFGS-B')
    if res.success:
        gp.length_scale = np.exp(res.x[0])
        gp.noise = np.exp(res.x[1])
        # Refit with updated hyperparams
        gp.fit(gp.X_train, gp.y_train)
    else:
        print("Hyperparameter optimization failed, using original parameters.")
    return gp.length_scale, gp.noise

########################################
# Main Script with argparse
########################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Active Learning with various acquisition functions.")
    parser.add_argument('--acquisition', type=str, default='ucb', choices=['ucb','std','ei','pi'],
                        help="Acquisition function: 'ucb','std','ei','pi'")
    parser.add_argument('--num_points_grid', nargs='+', type=int, default=[5,10,20],
                        help="List of num_points values to try.")
    parser.add_argument('--query_size', type=int, default=5,
                        help="How many points to add each iteration.")
    parser.add_argument('--length_scale', type=float, default=1.0,
                        help="Initial length scale.")
    parser.add_argument('--noise', type=float, default=0.1,
                        help="Initial noise level.")
    parser.add_argument('--random_seed', type=int, default=42,
                        help="Random seed for reproducibility.")

    args = parser.parse_args()

    # Load and Preprocess Data
    data = pd.read_csv("Main.csv")
    required_columns = ['P', 'VF', 'LCD', 'PLD', 'SA', 'y']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found.")
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data.dropna(subset=required_columns, inplace=True)
    data = data[(data['P']>0) & (data['y']>0)]

    X_original = data[['P','VF','LCD','PLD','SA']].values
    y_original = data['y'].values.reshape(-1,1)

    X_transformed = X_original.copy()
    X_transformed[:,0]=np.log10(X_transformed[:,0])
    y_transformed = np.log10(y_original)

    if np.isnan(X_transformed).any() or np.isinf(X_transformed).any():
        raise ValueError("NaN/Inf in X_transformed.")
    if np.isnan(y_transformed).any() or np.isinf(y_transformed).any():
        raise ValueError("NaN/Inf in y_transformed.")

    x_m = X_transformed.mean(axis=0)
    x_std = X_transformed.std(axis=0)
    x_std[x_std==0]=1e-6
    X_scaled = (X_transformed - x_m)/x_std

    y_m = y_transformed.mean()
    y_std = y_transformed.std()
    if y_std==0:
        y_std=1e-6
    y_scaled = (y_transformed - y_m)/y_std

    # Grid Search Over num_points
    results = []
    for np_val in args.num_points_grid:
        print(f"\nRunning AL for num_points={np_val} with acquisition={args.acquisition} and query_size={args.query_size}...")
        mean_var, sel_idx, gp_final = active_learning_run(X_scaled, y_scaled,
                                                          num_points=np_val,
                                                          acquisition=args.acquisition,
                                                          length_scale=args.length_scale,
                                                          noise=args.noise,
                                                          random_seed=args.random_seed,
                                                          query_size=args.query_size)
        results.append((np_val, mean_var, sel_idx, gp_final))
        print(f"For num_points={np_val}, Mean Var={mean_var:.6f}")

    results.sort(key=lambda x: x[1])
    best_config = results[0]

    print("\nBest configuration by lowest mean uncertainty:")
    print(f"  num_points={best_config[0]}, Mean Variance={best_config[1]:.6f}")
    print("Selected indices:", best_config[2])

    final_num_points = best_config[0]
    final_indices = best_config[2]
    gp_final = best_config[3]

    # Optimize hyperparameters on final chosen points
    opt_length_scale, opt_noise = optimize_hyperparameters(gp_final)
    print(f"Optimized length_scale={opt_length_scale:.4f}, noise={opt_noise:.4f}")

    # Now gp_final uses optimized parameters
    mu_final, var_final = gp_final.predict(X_scaled)
    mu_final_unscaled = mu_final*y_std + y_m
    y_final_exp = np.power(10, mu_final_unscaled)

    # Save selected points
    selected_points_df = pd.DataFrame(X_original[final_indices], columns=['P','VF','LCD','PLD','SA'])
    selected_points_df['Index'] = final_indices
    selected_points_df.to_csv("selected_points.csv", index=False)
    print("Selected points saved to selected_points.csv")

    # Save predictions
    predictions_df = pd.DataFrame(X_original, columns=['P','VF','LCD','PLD','SA'])
    predictions_df['Predicted_y_log'] = mu_final_unscaled
    predictions_df['Predicted_y_original'] = y_final_exp
    predictions_df['Var_log'] = var_final
    predictions_df.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")
