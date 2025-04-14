"""
##  OAI dataset
##  Causal Inference / Abnormal Region Detection
##  Bayesian Method
"""

# import os
# from os import mkdir
# import argparse
import numpy as np
# import random
# from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.decomposition import NMF
import seaborn as sns
# from sklearn.decomposition import PCA
# from functions import soft_threshold, update_sigma2, update_alpha, update_beta, update_gamma, update_tau, update_score1
# from scipy.ndimage import gaussian_filter
# from sklearn.feature_extraction import image
# from sklearn.cluster import spectral_clustering
import scipy.stats as stats
# from sklearn.linear_model import LogisticRegression
import pandas as pd
# from scipy.io import loadmat, savemat
# import sys
# import glob
# import os
# from zipfile import ZipFile

"""
import all required packages above
"""

""" directory settings """
real_data_path = './dat/real_data'
real_result_path = './dat/real_result'

# MCMC settings
n_iter = 2200     # total iteration number
n_burnin = 200    # iteration number for burn-in stage
n_slice = 20     # slicing number

""" load real data """
dx = np.load('%s/dx.npy' % real_data_path)  # dx. diagnosis information: normal control (0) and patient (>0)
x = np.load('%s/x.npy' % real_data_path)  # x. covariates: sex (F==1), bmi, age
y_ds_old = np.load('%s/y_ds.npy' % real_data_path)  # y_ds. (down-sampled) thickness map (n*m)

# Replace NaN with 0
y_ds = np.nan_to_num(y_ds_old, nan=0)  # nan=0 replaces NaN with 0

# # Check if the array contains any NaN values
# has_na = np.isnan(y_ds).any()
#
# print("Does the array have NA values?", has_na)

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
info_1 = np.hstack((x[dx[:, 0] >= 1, :], dx[dx[:, 0] >= 1, :]))
n_0 = x_0.shape[0]     # sample size for normal controls
n_1 = n - n_0     # sample size for oa patients
print([np.sum(1*(dx[:, 0]==0)), np.sum(1*(dx[:, 0]==1)), np.sum(1*(dx[:, 0]==2)), np.sum(1*(dx[:, 0]==3)), np.sum(1*(dx[:, 0]==4)), n])
print([np.sum(x[dx[:, 0]==0, 0]), np.sum(x[dx[:, 0]==1, 0]), np.sum(x[dx[:, 0]==2, 0]), np.sum(x[dx[:, 0]==3, 0]), np.sum(x[dx[:, 0]==4, 0])])
print([np.sum(1*(dx[:, 0]==0))-np.sum(x[dx[:, 0]==0, 0]),
       np.sum(1*(dx[:, 0]==1))-np.sum(x[dx[:, 0]==1, 0]),
       np.sum(1*(dx[:, 0]==2))-np.sum(x[dx[:, 0]==2, 0]),
       np.sum(1*(dx[:, 0]==3))-np.sum(x[dx[:, 0]==3, 0]),
       np.sum(1*(dx[:, 0]==4))-np.sum(x[dx[:, 0]==4, 0])])
print([[np.mean(x[dx[:, 0]==0, 1]), np.std(x[dx[:, 0]==0, 1])],
       [np.mean(x[dx[:, 0]==1, 1]), np.std(x[dx[:, 0]==1, 1])],
       [np.mean(x[dx[:, 0]==2, 1]), np.std(x[dx[:, 0]==2, 1])],
       [np.mean(x[dx[:, 0]==3, 1]), np.std(x[dx[:, 0]==3, 1])],
       [np.mean(x[dx[:, 0]==4, 1]), np.std(x[dx[:, 0]==4, 1])],
       [np.mean(x[:, 1]), np.std(x[:, 1])]])
print([[np.mean(x[dx[:, 0]==0, 2]), np.std(x[dx[:, 0]==0, 2])],
       [np.mean(x[dx[:, 0]==1, 2]), np.std(x[dx[:, 0]==1, 2])],
       [np.mean(x[dx[:, 0]==2, 2]), np.std(x[dx[:, 0]==2, 2])],
       [np.mean(x[dx[:, 0]==3, 2]), np.std(x[dx[:, 0]==3, 2])],
       [np.mean(x[dx[:, 0]==4, 2]), np.std(x[dx[:, 0]==4, 2])],
       [np.mean(x[:, 2]), np.std(x[:, 2])]])
# print(np.sum(1*(dx==0)))
# print(np.sum(1*(dx==1)))
# print(np.sum(1*(dx==2)))
# print(np.sum(1*(dx==3)))
# print(np.sum(1*(dx==4)))


img_ds_h, img_ds_w = [62, 62]  # (down-sampled) image size 62*62
idx_ds = np.load('%s/idx_ds.npy' % real_data_path)  # idx_ds. index (1D) for mask area (minus 1 s.t. initial with 0)
n_v = idx_ds.shape[0]  # number of pixels in mask area
w_adj = np.load('%s/W.npy' % real_data_path)  # Adjacency matrix (n_v*n_v)
n_v_neighbor = np.load('%s/n_v_neighbor.npy' % real_data_path)  # number of pixels in neighborhood (n_v*1)

# simulated disease region map visualization
bv_seq = np.load('%s/bv_seq.npy' % real_result_path)
bv_pr = np.mean(bv_seq[n_burnin:n_iter:n_slice, :, :], axis=0)
bv_p = 1*(bv_pr>=0.5)
bv_n_p = (bv_p @ w_adj.T) / (np.ones((n_1, 1)) @ n_v_neighbor.T)
alpha_seq = np.load('%s/alpha_seq.npy' % real_result_path)
alpha_ss = 1*(np.mean(1*(alpha_seq[n_burnin:n_iter:n_slice, :, :]==0), axis=0) < 0.5)
alpha_p = np.mean(alpha_seq[n_burnin:n_iter:n_slice, :, :], axis=0) * alpha_ss
beta_seq = np.load('%s/beta_seq.npy' % real_result_path)
beta_ss = 1*(np.mean(1*(beta_seq[n_burnin:n_iter:n_slice, :, :]==0), axis=0) < 0.5)
beta_p = np.mean(beta_seq[n_burnin:n_iter:n_slice, :, :], axis=0) * beta_ss
gamma_seq = np.load('%s/gamma_seq.npy' % real_result_path)
# print(gamma_seq)
gamma_p = np.mean(gamma_seq)
tau_seq = np.load('%s/tau_seq.npy' % real_result_path)
# print(tau_seq)
tau_p = np.mean(tau_seq)
sigma2_seq = np.load('%s/sigma2_seq.npy' % real_result_path)
sigma2_p = np.mean(sigma2_seq[n_burnin:n_iter:n_slice, :], axis=0)


# for j in range(n_1):
#     # simulated disease region map visualization
#     roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
#     roi_ds[0, idx_ds] = np.reshape(bv_p[j, :], shape=(n_v, 1))  # randomly select the 4-th subject
#     img_r = np.reshape(roi_ds, shape=(img_ds_h, img_ds_w))
#     # Display the array as an image
#     plt.imshow(img_r, cmap='viridis')
#     plt.colorbar()
#     plt.show()

# Plot the curve of gamma
roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(alpha_p[1, :], (n_v, 1))  # randomly select the 4-th subject
print([min(alpha_p[1, :]), max(alpha_p[1, :])])
alpha_1 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(alpha_p[2, :], (n_v, 1))  # randomly select the 4-th subject
alpha_2 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(alpha_p[3, :], (n_v, 1))  # randomly select the 4-th subject
alpha_3 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(alpha_p[4, :], (n_v, 1))  # randomly select the 4-th subject
alpha_4 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(alpha_p[5, :], (n_v, 1))  # randomly select the 4-th subject
alpha_5 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
# Plot each subplot and add captions
captions = ['Gender', 'BMI', 'Age', r'Gender$\times$BMI', r'Gender$\times$Age']  # Captions for each subplot
fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(10, 5))
norm = colors.TwoSlopeNorm(vmin=-0.02, vcenter=0, vmax=0.02)
im1 = axs[0].matshow(alpha_1, cmap='PuOr', norm=norm)
axs[0].axis('off')
axs[0].set_title(captions[0], fontsize=16)  # Set caption for each subplot
plt.colorbar(im1, ax=axs[0], shrink=0.3, aspect=20)  # Colorbar for the first subplot
im2 = axs[1].matshow(alpha_2, cmap='PuOr', norm=norm)
axs[1].axis('off')
axs[1].set_title(captions[1], fontsize=16)  # Set caption for each subplot
plt.colorbar(im2, ax=axs[1], shrink=0.3, aspect=20)  # Colorbar for the first subplot
im3 = axs[2].matshow(alpha_3, cmap='PuOr', norm=norm)
axs[2].axis('off')
axs[2].set_title(captions[2], fontsize=16)  # Set caption for each subplot
plt.colorbar(im3, ax=axs[2], shrink=0.3, aspect=20)  # Colorbar for the first subplot
im4 = axs[3].matshow(alpha_4, cmap='PuOr', norm=norm)
axs[3].axis('off')
axs[3].set_title(captions[3], fontsize=16)  # Set caption for each subplot
plt.colorbar(im4, ax=axs[3], shrink=0.3, aspect=20)  # Colorbar for the first subplot
im5 = axs[4].matshow(alpha_5, cmap='PuOr', norm=norm)
axs[4].axis('off')
axs[4].set_title(captions[4], fontsize=16)  # Set caption for each subplot
plt.colorbar(im5, ax=axs[4], shrink=0.3, aspect=20)  # Colorbar for the first subplot
# Adjust layout to leave space for colorbar
fig.subplots_adjust(right=0.95)  # Leave space on the right for colorbar
plt.show()

# Plot the curve of beta
roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(beta_p[1, :], (n_v, 1))  # randomly select the 4-th subject
beta_1 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(beta_p[2, :], (n_v, 1))  # randomly select the 4-th subject
beta_2 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(beta_p[3, :], (n_v, 1))  # randomly select the 4-th subject
beta_3 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(beta_p[4, :], (n_v, 1))  # randomly select the 4-th subject
beta_4 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(beta_p[5, :], (n_v, 1))  # randomly select the 4-th subject
beta_5 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(beta_p[6, :], (n_v, 1))  # randomly select the 4-th subject
beta_6 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
# Plot each subplot and add captions
captions = ['Gender', 'BMI', 'Age', r'Gender$\times$BMI', r'Gender$\times$Age', r'KLG']  # Captions for each subplot
fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(10, 5))
norm = colors.TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30)
im1 = axs[0].matshow(beta_1, cmap='PuOr', norm=norm)
axs[0].axis('off')
axs[0].set_title(captions[0], fontsize=16)  # Set caption for each subplot
plt.colorbar(im1, ax=axs[0], shrink=0.3, aspect=20)  # Colorbar for the first subplot
im2 = axs[1].matshow(beta_2, cmap='PuOr', norm=norm)
axs[1].axis('off')
axs[1].set_title(captions[1], fontsize=16)  # Set caption for each subplot
plt.colorbar(im2, ax=axs[1], shrink=0.3, aspect=20)  # Colorbar for the first subplot
im3 = axs[2].matshow(beta_3, cmap='PuOr', norm=norm)
axs[2].axis('off')
axs[2].set_title(captions[2], fontsize=16)  # Set caption for each subplot
plt.colorbar(im3, ax=axs[2], shrink=0.3, aspect=20)  # Colorbar for the first subplot
im4 = axs[3].matshow(beta_4, cmap='PuOr', norm=norm)
axs[3].axis('off')
axs[3].set_title(captions[3], fontsize=16)  # Set caption for each subplot
plt.colorbar(im4, ax=axs[3], shrink=0.3, aspect=20)  # Colorbar for the first subplot
im5 = axs[4].matshow(beta_5, cmap='PuOr', norm=norm)
axs[4].axis('off')
axs[4].set_title(captions[4], fontsize=16)  # Set caption for each subplot
plt.colorbar(im5, ax=axs[4], shrink=0.3, aspect=20)  # Colorbar for the first subplot
im6 = axs[5].matshow(beta_6, cmap='PuOr', norm=norm)
axs[5].axis('off')
axs[5].set_title(captions[5], fontsize=16)  # Set caption for each subplot
plt.colorbar(im6, ax=axs[5], shrink=0.3, aspect=20)  # Colorbar for the first subplot
# Adjust layout to leave space for colorbar
fig.subplots_adjust(right=0.95)  # Leave space on the right for colorbar
plt.show()



# Plot the curve of binary gamma/beta
roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(alpha_p[1, :], (n_v, 1))  # randomly select the 4-th subject
alpha_1 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(alpha_p[2, :], (n_v, 1))  # randomly select the 4-th subject
alpha_2 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(alpha_p[3, :], (n_v, 1))  # randomly select the 4-th subject
alpha_3 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(alpha_p[4, :], (n_v, 1))  # randomly select the 4-th subject
alpha_4 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(alpha_p[5, :], (n_v, 1))  # randomly select the 4-th subject
alpha_5 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(beta_p[1, :], (n_v, 1))  # randomly select the 4-th subject
beta_1 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(beta_p[2, :], (n_v, 1))  # randomly select the 4-th subject
beta_2 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(beta_p[3, :], (n_v, 1))  # randomly select the 4-th subject
beta_3 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(beta_p[4, :], (n_v, 1))  # randomly select the 4-th subject
beta_4 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(beta_p[5, :], (n_v, 1))  # randomly select the 4-th subject
beta_5 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
# Plot each subplot and add captions
captions = ['Gender', 'BMI', 'Age', r'Gender$\times$BMI', r'Gender$\times$Age']  # Captions for each subplot
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 5))
norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
im11 = axs[0,0].matshow(np.sign(alpha_1), cmap='PuOr', norm=norm)
axs[0,0].axis('off')
axs[0,0].set_title(captions[0], fontsize=16)  # Set caption for each subplot
#plt.colorbar(im11, ax=axs[0], shrink=0.3, aspect=20)  # Colorbar for the first subplot
im12 = axs[0,1].matshow(np.sign(alpha_2), cmap='PuOr', norm=norm)
axs[0,1].axis('off')
axs[0,1].set_title(captions[1], fontsize=16)  # Set caption for each subplot
#plt.colorbar(im12, ax=axs[1], shrink=0.3, aspect=20)  # Colorbar for the first subplot
im13 = axs[0,2].matshow(np.sign(alpha_3), cmap='PuOr', norm=norm)
axs[0,2].axis('off')
axs[0,2].set_title(captions[2], fontsize=16)  # Set caption for each subplot
#plt.colorbar(im13, ax=axs[2], shrink=0.3, aspect=20)  # Colorbar for the first subplot
im14 = axs[0,3].matshow(np.sign(alpha_4), cmap='PuOr', norm=norm)
axs[0,3].axis('off')
axs[0,3].set_title(captions[3], fontsize=16)  # Set caption for each subplot
#plt.colorbar(im14, ax=axs[3], shrink=0.3, aspect=20)  # Colorbar for the first subplot
im15 = axs[0,4].matshow(np.sign(alpha_5), cmap='PuOr', norm=norm)
axs[0,4].axis('off')
axs[0,4].set_title(captions[4], fontsize=16)  # Set caption for each subplot
#plt.colorbar(im15, ax=axs[0,4], shrink=0.3, aspect=20)  # Colorbar for the first subplot

im21 = axs[1,0].matshow(np.sign(beta_1), cmap='PuOr', norm=norm)
axs[1,0].axis('off')
#axs[5].set_title(captions[0], fontsize=16)  # Set caption for each subplot
#plt.colorbar(im21, ax=axs[5], shrink=0.3, aspect=20)  # Colorbar for the first subplot
im22 = axs[1,1].matshow(np.sign(beta_2), cmap='PuOr', norm=norm)
axs[1,1].axis('off')
#axs[1].set_title(captions[1], fontsize=16)  # Set caption for each subplot
#plt.colorbar(im22, ax=axs[1], shrink=0.3, aspect=20)  # Colorbar for the first subplot
im23 = axs[1,2].matshow(np.sign(beta_3), cmap='PuOr', norm=norm)
axs[1,2].axis('off')
#axs[2].set_title(captions[2], fontsize=16)  # Set caption for each subplot
#plt.colorbar(im23, ax=axs[2], shrink=0.3, aspect=20)  # Colorbar for the first subplot
im24 = axs[1,3].matshow(np.sign(beta_4), cmap='PuOr', norm=norm)
axs[1,3].axis('off')
#axs[3].set_title(captions[3], fontsize=16)  # Set caption for each subplot
#plt.colorbar(im24, ax=axs[3], shrink=0.3, aspect=20)  # Colorbar for the first subplot
im25 = axs[1,4].matshow(np.sign(beta_5), cmap='PuOr', norm=norm)
axs[1,4].axis('off')
#axs[4].set_title(captions[4], fontsize=16)  # Set caption for each subplot
plt.colorbar(im25, ax=axs[1,4], shrink=0.3, aspect=20)  # Colorbar for the first subplot

# Adjust layout to leave space for colorbar
fig.subplots_adjust(right=0.95)  # Leave space on the right for colorbar
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(gamma_seq, color='blue', linewidth=2)
plt.xlabel('iteration', fontsize=16)
plt.ylabel(r'$\gamma$', fontsize=16)
plt.title(r'Visualization of $\gamma$ sampling', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

# Plot the curve of tau
plt.figure(figsize=(8, 5))
plt.plot(tau_seq, color='blue', linewidth=2)
plt.xlabel('iteration', fontsize=16)
plt.ylabel(r'$\tau$', fontsize=16)
plt.title(r'Visualization of $\tau$ sampling', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

# Plot the curve of tau
plt.figure(figsize=(8, 5))
plt.plot(tau_seq, color='blue', linewidth=2)
plt.xlabel('iteration')
plt.ylabel(r'$\tau$')
plt.title(r'Visualization of $\tau$ sampling')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

# Plot the curve of sigma20
plt.figure(figsize=(8, 5))
plt.plot(sigma2_seq[:, 10], color='blue', linewidth=2)
plt.xlabel('iteration')
plt.ylabel(r'sigma2(s)')
plt.title(r'Visualization of Sigma20 sampling')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

# simulated disease region map visualization
roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(sigma2_p, (n_v, 1))  # randomly select the 4-th subject
img_r = np.reshape(roi_ds, (img_ds_h, img_ds_w))
# Display the array as an image
plt.imshow(img_r, cmap='viridis')
plt.colorbar()
plt.show()

# generate subgroup disease mapping
dx_1 = dx[dx[:, 0]>0, 0]
bv_pr_1 = np.mean(bv_pr[dx_1==1, :], axis=0)
roi_ds = -0.1*np.ones(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(bv_pr_1, (n_v, 1))  # randomly select the 4-th subject
klg_1 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
bv_pr_2 = np.mean(bv_pr[dx_1==2, :], axis=0)
roi_ds = -0.1*np.ones(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(bv_pr_2, (n_v, 1))  # randomly select the 4-th subject
klg_2 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
bv_pr_34 = np.mean(bv_pr[dx_1>=3, :], axis=0)
roi_ds = -0.1*np.ones(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(bv_pr_34, (n_v, 1))  # randomly select the 4-th subject
klg_34 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
bv_pr_all = np.mean(bv_pr, axis=0)
roi_ds = -0.1*np.ones(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(bv_pr_all, (n_v, 1))  # randomly select the 4-th subject
klg_all = np.reshape(roi_ds, (img_ds_h, img_ds_w))

captions = ['KLG 1', 'KLG 2', 'KLG 3 or 4', 'OA patients (KLG = 1, 2, 3 or 4)']  # Captions for each subplot
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(10, 5))
im1 = axs[0].matshow(klg_1, cmap='viridis', vmin=-0.01, vmax=0.2)
axs[0].axis('off')
axs[0].set_title(captions[0], fontsize=12)  # Set caption for each subplot
plt.colorbar(im1, ax=axs[0], shrink=0.3, aspect=20)  # Colorbar for the first subplot
im2 = axs[1].matshow(klg_2, cmap='viridis', vmin=-0.01, vmax=0.2)
axs[1].axis('off')
axs[1].set_title(captions[1], fontsize=12)  # Set caption for each subplot
plt.colorbar(im2, ax=axs[1], shrink=0.3, aspect=20)  # Colorbar for the first subplot
im34 = axs[2].matshow(klg_34, cmap='viridis', vmin=-0.01, vmax=0.2)
axs[2].axis('off')
axs[2].set_title(captions[2], fontsize=12)  # Set caption for each subplot
plt.colorbar(im34, ax=axs[2], shrink=0.3, aspect=20)  # Colorbar for the first subplot
im1234 = axs[3].matshow(klg_all, cmap='viridis', vmin=-0.01, vmax=0.2)
axs[3].axis('off')
axs[3].set_title(captions[3], fontsize=12)  # Set caption for each subplot
plt.colorbar(im1234, ax=axs[3], shrink=0.3, aspect=20)  # Colorbar for the first subplot
# Adjust layout to leave space for colorbar
fig.subplots_adjust(right=0.95)  # Leave space on the right for colorbar
plt.show()




# clustering based on disease regions
# Initialize NMF
n_components = 3  # Number of latent components
model = NMF(n_components=n_components, tol=5e-3)

# Fit the model
W = model.fit_transform(bv_pr)  # Basis matrix
H = model.components_       # Coefficients matrix

subgroup_indices = np.argmax(W, axis=1)
klg_1 = dx[dx[:, 0]>0, 0]
x_oa = x[dx[:, 0]>0, 0:3]
for k in range(n_components):
    klg_k = klg_1[subgroup_indices==k]
    print([np.sum(x_oa[subgroup_indices==k, 0], axis=0), np.sum(1*(x_oa[subgroup_indices==k, 0]==0)), np.mean(x_oa[subgroup_indices==k, 1], axis=0),
           np.std(x_oa[subgroup_indices==k, 1], axis=0), np.mean(x_oa[subgroup_indices==k, 2], axis=0),
           np.std(x_oa[subgroup_indices==k, 2], axis=0)])
    print([np.sum(1*(klg_k==1)), np.sum(1*(klg_k==2)), np.sum(1*(klg_k==3)), np.sum(1*(klg_k==4))])


roi_ds = -0.1*np.ones(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(H[0, :], (n_v, 1))  # randomly select the 4-th subject
group_1 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
roi_ds = -0.1*np.ones(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(H[1, :], (n_v, 1))  # randomly select the 4-th subject
group_2 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
roi_ds = -0.1*np.ones(shape=(1, img_ds_h * img_ds_w))
roi_ds[0, idx_ds] = np.reshape(H[2, :], (n_v, 1))  # randomly select the 4-th subject
group_3 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
# roi_ds = -0.1*np.ones(shape=(1, img_ds_h * img_ds_w))
# roi_ds[0, idx_ds] = np.reshape(H[3, :], (n_v, 1))  # randomly select the 4-th subject
# group_4 = np.reshape(roi_ds, (img_ds_h, img_ds_w))
# Plot each subplot and add captions
captions = ['Subgroup 1', 'Subgroup 2', 'Subgroup 3']  # Captions for each subplot
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
im1 = axs[0].matshow(group_1, cmap='viridis', vmin=-0.1, vmax=1)
axs[0].axis('off')
axs[0].set_title(captions[0], fontsize=12)  # Set caption for each subplot
plt.colorbar(im1, ax=axs[0], shrink=0.3, aspect=20)  # Colorbar for the first subplot
im2 = axs[1].matshow(group_2, cmap='viridis', vmin=-0.1, vmax=1)
axs[1].axis('off')
axs[1].set_title(captions[1], fontsize=12)  # Set caption for each subplot
plt.colorbar(im2, ax=axs[1], shrink=0.3, aspect=20)  # Colorbar for the first subplot
im3 = axs[2].matshow(group_3, cmap='viridis', vmin=-0.1, vmax=1)
axs[2].axis('off')
axs[2].set_title(captions[2], fontsize=12)  # Set caption for each subplot
plt.colorbar(im3, ax=axs[2], shrink=0.3, aspect=20)  # Colorbar for the first subplot
# im4 = axs[3].matshow(group_4, cmap='viridis')
# axs[3].axis('off')
# axs[3].set_title(captions[3], fontsize=12)  # Set caption for each subplot
# plt.colorbar(im4, ax=axs[3], shrink=0.3, aspect=20)  # Colorbar for the first subplot
# Adjust layout to leave space for colorbar
fig.subplots_adjust(right=0.95)  # Leave space on the right for colorbar
plt.show()


# disease region detection visualization
pcs = np.load('%s/pcs.npy' % real_result_path)
n_pcs = pcs.shape[0]
print(pcs.shape)
fig, axs = plt.subplots(nrows=6, ncols=5, figsize=(10, 5))
for j in range(6):
    for k in range(5):
        roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
        roi_ds[0, idx_ds] = np.reshape(pcs[:, j*5+k], (n_v, 1))  # randomly select the 4-th subject
        img_r = np.reshape(roi_ds, (img_ds_h, img_ds_w))
        im_jk = axs[j, k].matshow(img_r, cmap='viridis')
        axs[j, k].axis('off')
        axs[j, k].set_title('PC {}'.format(j*5+k+1), fontsize=12, pad=-3)  # Set caption for each subplot
        plt.colorbar(im1, ax=axs[j, k], shrink=0.5, aspect=20)  # Colorbar for the first subplot
fig.subplots_adjust(hspace=0.8, wspace=0.3)
# Adjust layout to leave space for colorbar
fig.subplots_adjust(right=0.95)  # Leave space on the right for colorbar
# Add a main caption at the top of the figure
plt.show()

# # disease region detection visualization
# for i in range(100):
#     roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
#     roi_ds[0, idx_ds] = np.reshape(y_1[i, :], (n_v, 1))  # randomly select the 4-th subject
#     img_r = np.reshape(roi_ds, (img_ds_h, img_ds_w))
#     roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
#     roi_ds[0, idx_ds] = np.reshape(y_1[i, :] + 5 * bv_p[i, :], (n_v, 1))  # randomly select the 4-th subject
#     bv_hat = np.reshape(roi_ds, (img_ds_h, img_ds_w))
#     roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
#     roi_ds[0, idx_ds] = np.reshape(bv_pr[i, :], (n_v, 1))  # randomly select the 4-th subject
#     bv_pr_hat = np.reshape(roi_ds, (img_ds_h, img_ds_w))
#     # Plot each subplot and add captions
#     captions = ['Thickness Map', 'Detected Disease Region', 'Probability']  # Captions for each subplot
#     fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
#     im1 = axs[0].matshow(img_r, cmap='viridis')
#     axs[0].axis('off')
#     axs[0].set_title(captions[0], fontsize=12)  # Set caption for each subplot
#     plt.colorbar(im1, ax=axs[0], shrink=0.2, aspect=20)  # Colorbar for the first subplot
#     im2 = axs[1].matshow(bv_hat, cmap='viridis')
#     axs[1].axis('off')
#     axs[1].set_title(captions[1], fontsize=12)  # Set caption for each subplot
#     plt.colorbar(im2, ax=axs[1], shrink=0.2, aspect=20)  # Colorbar for the first subplot
#     im3 = axs[2].matshow(bv_pr_hat, cmap='viridis')
#     axs[2].axis('off')
#     axs[2].set_title(captions[2], fontsize=12)  # Set caption for each subplot
#     plt.colorbar(im3, ax=axs[2], shrink=0.2, aspect=20)  # Colorbar for the first subplot
#     # Adjust layout to leave space for colorbar
#     fig.subplots_adjust(right=0.95)  # Leave space on the right for colorbar
#     # Add a main caption at the top of the figure
#     sex_i = 'Female' if info_1[i, 0] == 1 else 'Male'
#     bmi_i = info_1[i, 1]
#     age_i = info_1[i, 2]
#     klg_i = info_1[i, 3]
#     plt.subtitle(r'Patient: SEX={}, BMI={}, AGE={}, KLG={}'.format(sex_i, bmi_i, age_i, klg_i), fontsize=16)
#     # Adjust layout to make room for the caption
#     plt.tight_layout(rect=[0, 0, 1, 1.5])  # Leave space for subtitle
#     plt.show()

for i in range(100):
    roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
    roi_ds[0, idx_ds] = np.reshape(y_1[i, :], (n_v, 1))  # randomly select the 4-th subject
    img_r = np.reshape(roi_ds, (img_ds_h, img_ds_w))
    roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
    roi_ds[0, idx_ds] = np.reshape(y_1[i, :] + 5 * bv_p[i, :], (n_v, 1))  # randomly select the 4-th subject
    bv_hat = np.reshape(roi_ds, (img_ds_h, img_ds_w))
    roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
    roi_ds[0, idx_ds] = np.reshape(y_1[i, :]-x_1[i, :] @ alpha_p, (n_v, 1))  # randomly select the 4-th subject
    ite = np.reshape(roi_ds, (img_ds_h, img_ds_w))
    roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
    roi_ds[0, idx_ds] = np.reshape(gamma_p + tau_p * bv_n_p[i, :], (n_v, 1))  # randomly select the 4-th subject
    tse = np.reshape(roi_ds, (img_ds_h, img_ds_w))
    roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
    roi_ds[0, idx_ds] = np.reshape(tau_p * bv_n_p[i, :], (n_v, 1))  # randomly select the 4-th subject
    ise = np.reshape(roi_ds, (img_ds_h, img_ds_w))
    # Plot each subplot and add captions
    captions = ['Thickness Map', 'Detected Disease Region', 'ITE', 'TSE', 'ISE']  # Captions for each subplot
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(10, 5))
    im1 = axs[0].matshow(img_r, cmap='viridis')
    axs[0].axis('off')
    axs[0].set_title(captions[0], fontsize=12)  # Set caption for each subplot
    plt.colorbar(im1, ax=axs[0], shrink=0.2, aspect=20)  # Colorbar for the first subplot
    im2 = axs[1].matshow(bv_hat, cmap='viridis')
    axs[1].axis('off')
    axs[1].set_title(captions[1], fontsize=12)  # Set caption for each subplot
    plt.colorbar(im2, ax=axs[1], shrink=0.2, aspect=20)  # Colorbar for the first subplot
    im3 = axs[2].matshow(ite, cmap='viridis')
    axs[2].axis('off')
    axs[2].set_title(captions[2], fontsize=12)  # Set caption for each subplot
    plt.colorbar(im3, ax=axs[2], shrink=0.2, aspect=20)  # Colorbar for the first subplot
    im4 = axs[3].matshow(tse, cmap='viridis')
    axs[3].axis('off')
    axs[3].set_title(captions[3], fontsize=12)  # Set caption for each subplot
    plt.colorbar(im4, ax=axs[3], shrink=0.2, aspect=20)  # Colorbar for the first subplot
    im5 = axs[4].matshow(ise, cmap='viridis')
    axs[4].axis('off')
    axs[4].set_title(captions[4], fontsize=12)  # Set caption for each subplot
    plt.colorbar(im5, ax=axs[4], shrink=0.2, aspect=20)  # Colorbar for the first subplot
    # Adjust layout to leave space for colorbar
    fig.subplots_adjust(right=0.95)  # Leave space on the right for colorbar
    # Add a main caption at the top of the figure
    sex_i = 'Female' if info_1[i, 0] == 1 else 'Male'
    bmi_i = info_1[i, 1]
    age_i = info_1[i, 2]
    klg_i = info_1[i, 3]
    plt.suptitle(r'Patient: SEX={}, BMI={}, AGE={}, KLG={}'.format(sex_i, bmi_i, age_i, klg_i), fontsize=16)
    # Adjust layout to make room for the caption
    plt.tight_layout(rect=[0, 0, 1, 1.5])  # Leave space for subtitle
    plt.show()