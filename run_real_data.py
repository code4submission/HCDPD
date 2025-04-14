"""
##  OAI dataset
##  Causal Inference / Abnormal Region Detection
##  Bayesian Method
"""

import os
import random
from os import mkdir
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
from scipy.ndimage import gaussian_filter
import scipy.stats as stats
# from scipy.stats import gamma #, bernoulli
from sklearn.decomposition import PCA
# from sklearn.feature_extraction import image
# from sklearn.cluster import spectral_clustering
from functions import (soft_threshold, update_sigma2, update_alpha,
                       update_beta, update_gamma, update_tau, update_score0, update_score1)
import time


"""
import all required packages above
"""

print('Real data analysis...')
""" set random seed """
random.seed(500)

start_time = time.time()

""" directory settings """
real_data_path = './dat/real_data'
real_result_path = './dat/real_result'
if os.path.isdir(real_result_path) is False:
    mkdir(real_result_path)

""" load real data """
dx = np.load('%s/dx.npy' % real_data_path)  # dx. diagnosis information: normal control (0) and patient (>0)
x = np.load('%s/x.npy' % real_data_path)  # x. covariates: sex (F==1), bmi, age
y_ds_old = np.load('%s/y_ds.npy' % real_data_path)  # y_ds. (down-sampled) thickness map (n*n_v)
# Replace NaN with 0
y_ds = np.nan_to_num(y_ds_old, nan=0)  # nan=0 replaces NaN with 0

# Construct the design matrix
n = x.shape[0]
sex = x[:, 0]  # First column (sex)
bmi = stats.zscore(x[:, 1])  # Z-score normalization for BMI (second column)
age = stats.zscore(x[:, 2])  # Z-score normalization for age (third column)
sex_bmi = sex * bmi  # Z-score normalization for sex * age
sex_age = sex * age  # Z-score normalization for sex * age
x_mat = np.column_stack((np.ones(n), sex, bmi, age, sex_bmi, sex_age))
p = x_mat.shape[1]

y_0 = y_ds[dx[:, 0] == 0, :]  # Filter rows where dx == 0
x_0 = x_mat[dx[:, 0] == 0, :]
y_1 = y_ds[dx[:, 0] >= 1, :]  # Filter rows where dx == 1
x_1 = x_mat[dx[:, 0] >= 1, :]
n_0 = x_0.shape[0]     # sample size for normal controls
n_1 = n - n_0     # sample size for oa patients
g_info = np.reshape(dx[dx[:, 0] > 0, 0], (n_1, 1))

img_ds_h, img_ds_w = [62, 62]  # (down-sampled) image size 62*62
idx_ds = np.load('%s/idx_ds.npy' % real_data_path)  # idx_ds. index (1D) for mask area (minus 1 s.t. initial with 0)
n_v = idx_ds.shape[0]  # number of pixels in mask area
w_adj = np.load('%s/W.npy' % real_data_path)  # Adjacency matrix (n_v*n_v)
n_v_neighbor = np.load('%s/n_v_neighbor.npy' % real_data_path)  # number of pixels in neighborhood (n_v*1)

# Load the initial estimation of disease region
bv_ini = np.load('%s/bv_ini.npy' % real_result_path)  # disease region pattern (n_1*n_v)
# roi_ds = np.zeros(shape=(1, img_ds_h*img_ds_w))
# roi_ds[0, idx_ds] = 1
# img_ds = np.reshape(roi_ds, (img_ds_h, img_ds_w))
# mask = img_ds.astype(bool)
# for i in range(n_1):
#     roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
#     roi_ds[0, idx_ds] = np.reshape(bv_ini[i, :], (n_v, 1))  # randomly select the 4-th subject
#     img_r = np.reshape(roi_ds*10, (img_ds_h, img_ds_w))+mask
#     # Display the array as an image
#     plt.imshow(img_r, cmap='viridis')
#     # plt.colorbar()
#     plt.show()

print("Get initial estimation of alpha")
# First smoothing, then estimation
smoothed_y0 = y_0
for i in range(n_0):
    roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
    roi_ds[0, idx_ds] = np.reshape(y_0[i, :], (n_v, 1))
    img_2d = np.reshape(roi_ds, (img_ds_h, img_ds_w))
    # Apply Gaussian filter
    smoothed_img_2d = gaussian_filter(img_2d, sigma=1)
    smoothed_img_r = np.reshape(smoothed_img_2d, (1, img_ds_h * img_ds_w))
    smoothed_y0[i, :] = np.reshape(smoothed_img_r[0, idx_ds], (n_v,))
alpha_ini = soft_threshold(np.linalg.solve(x_0.T @ x_0, x_0.T @ smoothed_y0), 0.01)
for j in range(alpha_ini.shape[0]):
    roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
    roi_ds[0, idx_ds] = np.reshape(alpha_ini[j, :], (n_v, 1))  # randomly select the 4-th subject
    img_r = np.reshape(roi_ds*10, (img_ds_h, img_ds_w))
    # Display the array as an image
    plt.imshow(img_r, cmap='viridis')
    # plt.colorbar()
    plt.show()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

""" Get initial estimation of pcs, scores_0, and sigma0 """
smoothed_res_ini = smoothed_y0 - x_0 @ alpha_ini
# Perform PCA on the smoothed residuals
pca = PCA()
pca.fit(smoothed_res_ini)
pc_coeff = pca.components_.T  # PCA components (transposed for column-wise similarity)
pc_score = pca.fit_transform(smoothed_res_ini)
explained = pca.explained_variance_ratio_ * 100  # Explained variance percentage
cum_explained = np.cumsum(explained)  # Cumulative explained variance
# Select the top n_pcs principal components
print(cum_explained[:30])
# scores_0 = pc_score[:, :20]
# print(np.std(scores_0, axis=0))
n_pcs = 30     # explain above 80% variance of y
pcs = pc_coeff[:, :n_pcs]
scores_0_ini = pc_score[:, :n_pcs]


# Calculate bv_n (scaled bv using w_adj and neighboring vertices)
bv_n_ini = (bv_ini @ w_adj.T) / (np.ones((n_1, 1)) @ n_v_neighbor.T)

""" set initial values for gamma and tau """
gamma_ini = random.uniform(-0.6, -0.4)
tau_ini = random.uniform(-0.3, -0.1)

""" set the initial values of score_1 """
res_1 = y_1 - x_1 @ alpha_ini - gamma_ini * bv_ini - tau_ini * bv_n_ini
smoothed_res_1 = np.zeros(res_1.shape)
for i in range(n_1):
    roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
    roi_ds[0, idx_ds] = np.reshape(res_1[i, :], (n_v, 1))
    img_2d = np.reshape(roi_ds, (img_ds_h, img_ds_w))
    # Apply Gaussian filter
    smoothed_img_2d = gaussian_filter(img_2d, sigma=1)
    smoothed_img_r = np.reshape(smoothed_img_2d, (1, img_ds_h * img_ds_w))
    smoothed_res_1[i, :] = np.reshape(smoothed_img_r[0, idx_ds], (n_v,))
scores_1_ini = smoothed_res_1 @ pcs

""" set hyperparameters """
a_0 = 8
b_0 = 10  # Equivalent of floor(1 / ((a_0 - 1) * sigma20_mean)), calculation omitted
c_0 = 1
d_0 = 1
# lambda2_beta = 1 / gamma.rvs(a_0, scale=1 / b_0)
# nu_beta = np.vstack((np.ones(n_v), 0.8 - 0.6 * (beta_ini[1:p, :]==0)))
sigma2_gamma = 0.1
sigma2_tau = 0.1
p_threshold = 0.5

""" MCMC iteration settings """
n_iter = 2200    # total iteration number
sigma2_seq = np.zeros((n_iter, n_v))
alpha_seq = np.zeros((n_iter, p, n_v))
beta_seq = np.zeros((n_iter, p+2, n_v))
lambda2_alpha_seq = np.zeros((n_iter, 1))
gamma_seq = np.zeros((n_iter, 1))
tau_seq = np.zeros((n_iter, 1))
bv_seq = np.zeros((n_iter, n_1, n_v))
scores_0_seq = np.zeros((n_iter, n_0, n_pcs))
scores_1_seq = np.zeros((n_iter, n_1, n_pcs))

alpha_tt = alpha_ini
scores_0_tt = scores_0_ini
scores_1_tt = scores_1_ini
lambda2_alpha_tt = 0.1 # 1 / gamma.rvs(a_0, scale=1 / b_0)
nu_alpha_tt = np.vstack((np.ones(n_v), 0.8 - 0.6 * (alpha_ini[1:p, :] == 0)))
gamma_tt = gamma_ini
tau_tt = tau_ini
bv_tt = bv_ini
bv_n_tt = bv_n_ini

for tt in range(n_iter):
    if np.remainder(tt + 1, 100) == 0:
        print(f'iteration {tt + 1}')
        print(f'iteration {tt + 1}')
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")
    # update sigma20
    res_tt = y_0 - x_0 @ alpha_tt - scores_0_tt @ pcs.T
    sigma2_tt = update_sigma2(res_tt, a_0, b_0)
    sigma2_seq[tt, :] = np.reshape(sigma2_tt, (n_v,))

    # update alpha & z_alpha
    alpha_n_tt = (alpha_tt @ w_adj.T) / (np.ones((p, 1)) @ n_v_neighbor.T)
    alpha_tt = update_alpha(x_0, res_tt, alpha_tt, alpha_n_tt, sigma2_tt, lambda2_alpha_tt, nu_alpha_tt)
    alpha_seq[tt, :, :] = alpha_tt
    z_alpha_tt = 1 * (alpha_tt != 0)

    # # update lambda2_alpha
    # alpha_n_tt = (alpha_tt @ w_adj.T) / (np.ones((p, 1)) @ n_v_neighbor.T)
    # alpha_s = (alpha_tt - alpha_n_tt) ** 2 / (np.ones((p, 1)) @ np.reshape(sigma2_tt, (1, n_v)))
    # lambda2_alpha_tt = update_lambda2_alpha(a_0, b_0, z_alpha_tt, alpha_s)
    # lambda2_alpha_seq[tt, 0] = lambda2_alpha_tt

    # # update nu_alpha
    # nu_alpha_tt = update_nu_alpha(c_0, d_0, z_alpha_tt)

    # update gamma
    res_1_tt = y_1 - x_1 @ alpha_tt - tau_tt * bv_n_tt - scores_1_tt @ pcs.T
    gamma_tt = update_gamma(res_1_tt, bv_tt, sigma2_tt, gamma_tt, sigma2_gamma)
    gamma_seq[tt, 0] = gamma_tt

    # update tau
    res_2_tt = y_1 - x_1 @ alpha_tt - gamma_tt * bv_tt - scores_1_tt @ pcs.T
    tau_tt = update_tau(res_2_tt, bv_n_tt, sigma2_tt, tau_tt, sigma2_tau)
    tau_seq[tt, 0] = tau_tt

    # update beta, bv, and bv_n
    res_3_b0 = y_1 - x_1 @ alpha_tt - tau_tt * bv_n_tt - scores_1_tt @ pcs.T
    res_3_b1 = y_1 - x_1 @ alpha_tt - gamma_tt * bv_tt - tau_tt * bv_n_tt - scores_1_tt @ pcs.T
    beta_tt, bv_tt = update_beta(x_1, g_info, bv_tt, bv_n_tt, res_3_b0, res_3_b1, sigma2_tt, p_threshold)
    bv_n_tt = (bv_tt @ w_adj.T) / (np.ones((n_1, 1)) @ n_v_neighbor.T)
    beta_seq[tt, :, :] = beta_tt
    bv_seq[tt, :, :] = bv_tt

    # update score_0
    res_0 = smoothed_y0 - x_0 @ alpha_tt
    scores_0_tt = update_score0(res_0, pcs)
    scores_0_seq[tt, :, :] = scores_0_tt

    # update score_1
    res_1 = y_1 - x_1 @ alpha_tt - gamma_tt * bv_tt - tau_tt * bv_n_tt
    scores_1_tt = update_score1(res_1, img_ds_h, img_ds_w, idx_ds, pcs)
    scores_1_seq[tt, :, :] = scores_1_tt

# save all the MCMC samples
np.save('%s/sigma2_seq.npy' % real_result_path, sigma2_seq)
np.save('%s/alpha_seq.npy' % real_result_path, alpha_seq)
np.save('%s/beta_seq.npy' % real_result_path, beta_seq)
np.save('%s/gamma_seq.npy' % real_result_path, gamma_seq)
np.save('%s/tau_seq.npy' % real_result_path, tau_seq)
np.save('%s/bv_seq.npy' % real_result_path, bv_seq)
np.save('%s/pcs.npy' % real_result_path, pcs)
np.save('%s/scores_0_seq.npy' % real_result_path, scores_0_seq)
np.save('%s/scores_1_seq.npy' % real_result_path, scores_1_seq)
