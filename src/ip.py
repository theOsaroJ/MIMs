#!/usr/bin/env python3
import os
os.environ['MPLCONFIGDIR'] = '/tmp/'  # For matplotlib if needed

import matplotlib
matplotlib.use('Agg')  # Non-GUI backend

import torch
import numpy as np
import pandas as pd
import random
import time

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# ------------------------------
# Early Stopping
# ------------------------------
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
                return True
            return False

# ------------------------------
# Combined Kernel (RBF + RQ + Matern3/2)
# ------------------------------
def rbf_kernel(A, B, ls):
    d2 = torch.cdist(A,B)**2
    return torch.exp(-0.5*d2/(ls**2))

def rq_kernel(A, B, ls, alpha=1.0):
    d2 = torch.cdist(A,B)**2
    factor = 1.0 + d2/(2.0*alpha*ls**2)
    return factor**(-alpha)

def matern32_kernel(A, B, ls):
    r = torch.cdist(A,B)
    sqrt3 = np.sqrt(3)
    scaled_r = sqrt3*r/ls
    return (1.0+scaled_r)*torch.exp(-scaled_r)

def combined_kernel(A, B, ls, alpha_rq=1.0):
    K_rbf    = rbf_kernel(A, B, ls)
    K_rq     = rq_kernel(A, B, ls, alpha=alpha_rq)
    K_matern = matern32_kernel(A, B, ls)
    return (K_rbf + K_rq + K_matern)/3.0

# ------------------------------
# Inducing GP
# ------------------------------
class InducingGP(torch.nn.Module):
    def __init__(self, X, Y, X_orig,
                 num_inducing=200,
                 x_mean=None,
                 x_std=None,
                 alpha_rq=1.0):
        """
        X, Y: scaled data => shape [N,D], [N,1]
        X_orig: original data => shape [N,D]
        """
        super().__init__()
        self.X = X
        self.Y = Y
        self.X_orig = X_orig
        self.num_inducing = num_inducing
        self.x_mean = x_mean
        self.x_std  = x_std
        self.alpha_rq = alpha_rq

        # min/max in scaled domain
        self.x_min = self.X.min(dim=0).values
        self.x_max = self.X.max(dim=0).values

        # init Inducing
        self._init_inducing()

        # hyperparams
        self.ls    = torch.nn.Parameter(torch.tensor(1.0, device=device))
        self.noise = torch.nn.Parameter(torch.tensor(0.1, device=device))

    def _init_inducing(self):
        # K-means in CPU
        X_np = self.X.cpu().numpy()
        # optional sub-sample if huge
        if X_np.shape[0]>50000:
            idx_sub = np.random.choice(X_np.shape[0], 50000, replace=False)
            X_km = X_np[idx_sub]
        else:
            X_km = X_np

        kmeans = KMeans(n_clusters=self.num_inducing, n_init=10, random_state=42)
        kmeans.fit(X_km)
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
        # clamp
        centers = torch.clamp(centers, min=self.x_min, max=self.x_max)

        # find nearest y
        # invert scale
        c_unscaled = centers.cpu().numpy()*self.x_std + self.x_mean
        c_unscaled_fix = c_unscaled.copy()
        c_unscaled_fix[:,0] = np.power(10, c_unscaled_fix[:,0]) # invert log10 for 'P'

        distmat = cdist(c_unscaled_fix, self.X_orig)
        near_idx= np.argmin(distmat, axis=1)
        y_init  = self.Y[near_idx]

        self.inducing_x = torch.nn.Parameter(centers.clone().detach())  # [M,D]
        self.inducing_y = torch.nn.Parameter(y_init.clone().detach())    # [M,1]

    def clamp_inducing(self):
        with torch.no_grad():
            self.inducing_x.clamp_(min=self.x_min, max=self.x_max)

    def forward(self, Xp):
        """
        Predictive mean,var at scaled Xp
        """
        # K_mm
        K_mm = combined_kernel(self.inducing_x, self.inducing_x,
                               self.ls, alpha_rq=self.alpha_rq)
        K_mm = K_mm + 1e-6*torch.eye(K_mm.shape[0],device=device)
        L_mm = torch.cholesky(K_mm)

        # cross
        K_nm = combined_kernel(Xp, self.inducing_x,
                              self.ls, alpha_rq=self.alpha_rq)

        alpha = torch.cholesky_solve(self.inducing_y, L_mm) # [M,1]

        mu = K_nm @ alpha # [N,1]

        # var => diag(k_xx - k_xm K_mm^-1 k_mx) + noise^2
        # each kernel => k(x,x)=1 => but average => ~1
        diag_kxx = torch.ones(Xp.shape[0], device=device)
        v = torch.cholesky_solve(K_nm.transpose(0,1), L_mm) # [M,N]
        Q_xx = (K_nm * v.transpose(0,1)).sum(dim=1)
        var = diag_kxx - Q_xx + self.noise**2
        var= var.clamp_min(1e-8)
        return mu, var.unsqueeze(1)

    def NLL(self, Xb, Yb):
        """
        Negative Log-likelihood on a batch => approximate
        """
        mu,var= self.forward(Xb)
        resid= Yb.squeeze()- mu.squeeze()
        nll= 0.5*torch.log(2*np.pi*var) + 0.5*(resid**2)/var
        return nll.mean()

# ------------------------------
# Setup
# ------------------------------
seed=42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# ------------------------------
# Data Loading
# ------------------------------
df= pd.read_csv("Main.csv")
req_cols= ['P','VF','LCD','PLD','SA','y']
for c in req_cols:
    if c not in df.columns:
        raise ValueError(f"Missing {c} in df")
    df[c]= pd.to_numeric(df[c], errors='coerce')

df.dropna(subset=req_cols,inplace=True)
df= df[(df['P']>0) & (df['y']>0)]
print("Data shape after filtering:", df.shape)

Xo= df[['P','VF','LCD','PLD','SA']].values
yo= df['y'].values.reshape(-1,1)

# log transform
X_t = Xo.copy()
X_t[:,0]= np.log10(X_t[:,0])
y_t = np.log10(yo)

# check
if np.isnan(X_t).any() or np.isinf(X_t).any():
    raise ValueError("Inf/NaN in X_t")
if np.isnan(y_t).any() or np.isinf(y_t).any():
    raise ValueError("Inf/NaN in y_t")

x_mean= X_t.mean(axis=0)
x_std = X_t.std(axis=0)
y_mean= y_t.mean()
y_std = y_t.std()

x_std[x_std==0]=1e-6
if y_std==0:
    y_std=1e-6

X_s= (X_t - x_mean)/x_std
y_s= (y_t - y_mean)/y_std

# train-val
X_tr, X_va, y_tr, y_va, Xo_tr, Xo_va, yo_tr, yo_va= train_test_split(
    X_s,y_s, Xo,yo, test_size=0.2, random_state=seed
)

X_tr_t= FloatTensor(X_tr)
y_tr_t= FloatTensor(y_tr)
X_va_t= FloatTensor(X_va)
y_va_t= FloatTensor(y_va)

# ------------------------------
# Hyperparams
# ------------------------------
batch_size=2000
num_inducing=5000
lr=1e-3
epochs=200
alpha_rq=1.0

# Prepare DataLoaders
from torch.utils.data import TensorDataset, DataLoader
dset_tr= TensorDataset(X_tr_t, y_tr_t)
dl_tr= DataLoader(dset_tr, batch_size=batch_size, shuffle=True)

# Build model
gp= InducingGP(
    X=X_tr_t, Y=y_tr_t, X_orig=Xo_tr,
    num_inducing=num_inducing,
    x_mean=x_mean, x_std=x_std,
    alpha_rq=alpha_rq
).to(device)

opt= torch.optim.Adam(gp.parameters(), lr=lr)
stopper= EarlyStopping(patience=5, min_delta=1e-4)

# ------------------------------
# Train Loop
# ------------------------------
for ep in range(epochs):
    gp.train()
    losses=[]
    for xb, yb in dl_tr:
        opt.zero_grad()
        nll= gp.NLL(xb, yb)
        nll.backward()
        opt.step()
        gp.clamp_inducing()
        losses.append(nll.item())
    avg_tr_loss= np.mean(losses)

    # val
    gp.eval()
    with torch.no_grad():
        val_nll= gp.NLL(X_va_t, y_va_t).item()

    print(f"Epoch {ep+1}/{epochs} => trNLL={avg_tr_loss:.6f}, vaNLL={val_nll:.6f}")

    if stopper(val_nll):
        print("Early stopping at epoch:", ep+1)
        break

# ------------------------------
# Evaluate on Validation
# ------------------------------
gp.eval()
with torch.no_grad():
    mu_va, var_va= gp.forward(X_va_t)
    mu_va_np= mu_va.cpu().numpy()*y_std + y_mean
    var_va_np= var_va.cpu().numpy()*(y_std**2)
    yva_np= y_va_t.cpu().numpy()*y_std + y_mean

# Convert back from log space
max_exp=38
yva_exp= np.power(10, np.clip(yva_np, None, max_exp))
muva_exp= np.power(10, np.clip(mu_va_np, None, max_exp))

mae_val= mean_absolute_error(yva_exp, muva_exp)
r2_val= r2_score(yva_exp, muva_exp)
print(f"Validation => MAE={mae_val:.4f}, R2={r2_val:.4f}")

# ------------------------------
# Retrain on entire dataset
# ------------------------------
# Combine train+val
X_full= torch.cat((X_tr_t, X_va_t),dim=0)
y_full= torch.cat((y_tr_t, y_va_t),dim=0)
Xo_full= np.vstack((Xo_tr, Xo_va))
yo_full= np.vstack((yo_tr, yo_va))

full_gp= InducingGP(
    X=X_full, Y=y_full,
    X_orig=Xo_full,
    num_inducing=num_inducing,
    x_mean=x_mean, x_std=x_std,
    alpha_rq=alpha_rq
).to(device)

optF= torch.optim.Adam(full_gp.parameters(), lr=lr)
stopperF= EarlyStopping(patience=5, min_delta=1e-4)

# Make DataLoader
dset_full= TensorDataset(X_full, y_full)
dl_full= DataLoader(dset_full, batch_size=batch_size, shuffle=True)

for ep in range(epochs):
    full_gp.train()
    losses=[]
    for xb, yb in dl_full:
        optF.zero_grad()
        nll= full_gp.NLL(xb, yb)
        nll.backward()
        optF.step()
        full_gp.clamp_inducing()
        losses.append(nll.item())
    avg_loss= np.mean(losses)
    print(f"Final train epoch {ep+1}/{epochs}, NLL={avg_loss:.6f}")
    if stopperF(avg_loss):
        print("EarlyStop at epoch:", ep+1)
        break

# Evaluate on entire data
full_gp.eval()
with torch.no_grad():
    mu_all, var_all= full_gp.forward(X_full)
    mu_all_np= mu_all.cpu().numpy()*y_std + y_mean
    var_all_np= var_all.cpu().numpy()*(y_std**2)
    y_all_np= y_full.cpu().numpy()*y_std + y_mean

y_all_exp = np.power(10, np.clip(y_all_np, None, max_exp))
mu_all_exp= np.power(10, np.clip(mu_all_np, None, max_exp))
sigma_all_exp= np.power(10, np.clip(np.sqrt(var_all_np), None, max_exp))

mae_all= mean_absolute_error(y_all_exp, mu_all_exp)
r2_all= r2_score(y_all_exp, mu_all_exp)
print(f"Final on entire data => MAE={mae_all:.4f}, R2={r2_all:.4f}")

# ------------------------------
# Save CSV: GP_predictions.csv
# ------------------------------
# We replicate your approach:
# columns => 'P','VF','LCD','PLD','SA','True_y','Predicted_y','Sigma'
# invert scale for 'P'
P_unlog= np.power(10, X_full[:,0].cpu().numpy()*x_std[0] + x_mean[0])

df_pred= pd.DataFrame({
    'P': P_unlog,
    'VF': Xo_full[:,1],
    'LCD': Xo_full[:,2],
    'PLD': Xo_full[:,3],
    'SA':  Xo_full[:,4],
    'True_y': y_all_np.squeeze(),
    'Predicted_y': mu_all_np.squeeze(),
    'Sigma': np.sqrt(var_all_np).squeeze()
})

# Exponentiate y for final columns (cap at 10^38)
df_pred['True_y']      = np.power(10, np.clip(df_pred['True_y'], None, max_exp))
df_pred['Predicted_y'] = np.power(10, np.clip(df_pred['Predicted_y'], None, max_exp))
df_pred['Sigma']       = np.power(10, np.clip(df_pred['Sigma'], None, max_exp))

df_pred.to_csv("GP_predictions.csv", index=False)
print("Saved GP_predictions.csv")

# ------------------------------
# Save Inducing Points with Nearest
# ------------------------------
ind_x = full_gp.inducing_x.detach().cpu().numpy()
# unscale
ind_x_unscaled = ind_x * x_std + x_mean
# fix log for P
ind_x_unscaled_fix= ind_x_unscaled.copy()
ind_x_unscaled_fix[:,0] = np.power(10, ind_x_unscaled_fix[:,0])
# unscale y
ind_y= full_gp.inducing_y.detach().cpu().numpy()*y_std + y_mean
ind_y_unlog= np.power(10, np.clip(ind_y, None, max_exp))

df_ind= pd.DataFrame(ind_x_unscaled_fix, columns=['P','VF','LCD','PLD','SA'])
df_ind['y']= ind_y_unlog.squeeze()

# nearest data in original space
distmat= cdist(ind_x_unscaled_fix, Xo_full)
nearest_idx= np.argmin(distmat, axis=1)
nearest_pts= Xo_full[nearest_idx] # [M,5]
nearest_y= yo_full[nearest_idx]   # [M,1]

df_ind['Nearest_P']= nearest_pts[:,0]
df_ind['Nearest_VF']= nearest_pts[:,1]
df_ind['Nearest_LCD']= nearest_pts[:,2]
df_ind['Nearest_PLD']= nearest_pts[:,3]
df_ind['Nearest_SA']= nearest_pts[:,4]
df_ind['Nearest_y']= nearest_y.squeeze()

df_ind.to_csv("inducing_points_with_nearest_data.csv", index=False)
print("Saved inducing_points_with_nearest_data.csv")

print("All done.")
