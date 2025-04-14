"""
##  Bayesian Causal Inference / Heterogeneous Causal
Framework for Disease Pattern Detection
##  OAI dataset
## Report simulation studies results
## command line: python report_simu_data.py --id 1   (1 is the simulation id for random seed specification)
"""

# import os
# from os import mkdir
# import argparse
import numpy as np
# import random
# from scipy.spatial.distance import cdist
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.metrics import accuracy_score

# from sklearn.decomposition import PCA
# from functions import soft_threshold, update_sigma2, update_alpha, update_beta, update_gamma, update_tau, update_score1
# from scipy.ndimage import gaussian_filter
# from sklearn.feature_extraction import image
# from sklearn.cluster import spectral_clustering
# import scipy.stats as stats
# from sklearn.linear_model import LogisticRegression
# import pandas as pd
# from scipy.io import loadmat, savemat
# import sys
# import glob
# import os
# from zipfile import ZipFile

"""
import all required packages above
"""

def main():
    """
        Command-line function that takes input arguments.
        """
    parser = argparse.ArgumentParser(description="Run Causal Inference / Abnormal Region Detection on Simulation Datasets.")
    parser.add_argument("--id", type=int, required=True, help="Simulation ID")
    args = parser.parse_args()

    print('Simulation dataset %s.' % args.id)

    """ directory settings """
    real_data_path = './dat/real_data'
    simu_id = args.id
    simu_data_path = './dat/simu_data/simu_%s' % simu_id
    simu_result_path = './dat/simu_result/simu_%s' % simu_id

    # MCMC settings
    n_iter = 550     # total iteration number
    n_burnin = 50    # iteration number for burn-in stage
    n_slice =  5     # slicing number

    img_ds_h, img_ds_w = [62, 62]               # (down-sampled) image size 62*62
    idx_ds = np.load('%s/idx_ds.npy' % real_data_path)     # idx_ds. index (1D) for mask area (minus 1 s.t. initial with 0)
    n_v = idx_ds.shape[0]                           # number of pixels in mask area
    w_adj = np.load('%s/W.npy' % real_data_path)       # Adjacency matrix (n_v*n_v)
    n_v_neighbor = np.load('%s/n_v_neighbor.npy' % real_data_path)    # number of pixels in neighborhood (n_v*1)

    # # mask image visualization
    # roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
    # roi_ds[0, idx_ds] = 5  # randomly select the 4-th subject
    # img_r = np.reshape(roi_ds, shape=(img_ds_h, img_ds_w))
    # # Display the array as an image
    # fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    # im0 = axs.imshow(img_r, cmap='viridis')
    # axs.axis('off')
    # axs.set_title("Mask", fontsize=12)  # Set caption for each subplot
    # # plt.colorbar(im0, shrink=0.3, aspect=20)  # Colorbar for the first subplot
    # plt.show()

    # simulated disease region map visualization
    bv = np.load('%s/bv.npy' % simu_data_path)
    y_1 = np.load('%s/y_1.npy' % simu_data_path)
    x_1 = np.load('%s/x_1.npy' % simu_data_path)
    n_1 = x_1.shape[0]
    bv_seq = np.load('%s/bv_seq.npy' % simu_result_path)
    bv_p = 1*(np.mean(bv_seq[n_burnin:n_iter:n_slice, :, :], axis=0)>=0.5)
    bv_n_p = (bv_p @ w_adj.T) / (np.ones((n_1, 1)) @ n_v_neighbor.T)
    alpha_seq = np.load('%s/alpha_seq.npy' % simu_result_path)
    alpha_p = np.mean(alpha_seq[n_burnin:n_iter:n_slice, :, :], axis=0)
    z_alpha = 1*(np.abs(alpha_p)<=0.01)
    alpha_0 = np.load('%s/alpha0.npy' % simu_data_path)
    z_alpha_0 = 1*(alpha_0==0)
    z_acc = accuracy_score(z_alpha.flatten(), z_alpha_0.flatten())
    print(z_acc)

    beta_seq = np.load('%s/beta_seq.npy' % simu_result_path)
    beta_p = np.mean(beta_seq[n_burnin:n_iter:n_slice, :, :], axis=0)
    print([np.min(alpha_p), np.max(alpha_p)])
    gamma_seq = np.load('%s/gamma_seq.npy' % simu_result_path)
    gamma_p = np.mean(gamma_seq)
    tau_seq = np.load('%s/tau_seq.npy' % simu_result_path)
    tau_p = np.mean(tau_seq)
    sigma2_seq = np.load('%s/sigma2_seq.npy' % simu_result_path)
    # sigma2_p = np.mean(sigma2_seq[n_burnin:n_iter:n_slice, :], axis=0)

    # Plot the curve of gamma
    plt.figure(figsize=(8, 5))
    plt.plot(gamma_seq, color='blue', linewidth=2)
    plt.xlabel('iteration')
    plt.ylabel('gamma')
    plt.title('Visualization of Gamma sampling')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

    # Plot the curve of tau
    plt.figure(figsize=(8, 5))
    plt.plot(tau_seq, color='blue', linewidth=2)
    plt.xlabel('iteration')
    plt.ylabel('tau')
    plt.title('Visualization of Tau sampling')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

    # Plot the curve of sigma20
    plt.figure(figsize=(8, 5))
    plt.plot(sigma2_seq[:, 10], color='blue', linewidth=2)
    plt.xlabel('iteration')
    plt.ylabel('tau')
    plt.title('Visualization of Sigma20 sampling')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

    # Plot the curve of sigma20
    plt.figure(figsize=(8, 5))
    plt.plot(beta_seq[:, 1, 1000], color='blue', linewidth=2)
    plt.xlabel('iteration')
    plt.ylabel('Beta')
    plt.title('Visualization of Beta sampling')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

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
    norm = colors.TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=0.1)
    im1 = axs[0].matshow(alpha_1, cmap='PuOr', norm=norm)
    axs[0].axis('off')
    axs[0].set_title(captions[0], fontsize=12)  # Set caption for each subplot
    plt.colorbar(im1, ax=axs[0], shrink=0.3, aspect=20)  # Colorbar for the first subplot
    im2 = axs[1].matshow(alpha_2, cmap='PuOr', norm=norm)
    axs[1].axis('off')
    axs[1].set_title(captions[1], fontsize=12)  # Set caption for each subplot
    plt.colorbar(im2, ax=axs[1], shrink=0.3, aspect=20)  # Colorbar for the first subplot
    im3 = axs[2].matshow(alpha_3, cmap='PuOr', norm=norm)
    axs[2].axis('off')
    axs[2].set_title(captions[2], fontsize=12)  # Set caption for each subplot
    plt.colorbar(im3, ax=axs[2], shrink=0.3, aspect=20)  # Colorbar for the first subplot
    im4 = axs[3].matshow(alpha_4, cmap='PuOr', norm=norm)
    axs[3].axis('off')
    axs[3].set_title(captions[3], fontsize=12)  # Set caption for each subplot
    plt.colorbar(im4, ax=axs[3], shrink=0.3, aspect=20)  # Colorbar for the first subplot
    im5 = axs[4].matshow(alpha_5, cmap='PuOr', norm=norm)
    axs[4].axis('off')
    axs[4].set_title(captions[4], fontsize=12)  # Set caption for each subplot
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
    axs[0].set_title(captions[0], fontsize=12)  # Set caption for each subplot
    plt.colorbar(im1, ax=axs[0], shrink=0.3, aspect=20)  # Colorbar for the first subplot
    im2 = axs[1].matshow(beta_2, cmap='PuOr', norm=norm)
    axs[1].axis('off')
    axs[1].set_title(captions[1], fontsize=12)  # Set caption for each subplot
    plt.colorbar(im2, ax=axs[1], shrink=0.3, aspect=20)  # Colorbar for the first subplot
    im3 = axs[2].matshow(beta_3, cmap='PuOr', norm=norm)
    axs[2].axis('off')
    axs[2].set_title(captions[2], fontsize=12)  # Set caption for each subplot
    plt.colorbar(im3, ax=axs[2], shrink=0.3, aspect=20)  # Colorbar for the first subplot
    im4 = axs[3].matshow(beta_4, cmap='PuOr', norm=norm)
    axs[3].axis('off')
    axs[3].set_title(captions[3], fontsize=12)  # Set caption for each subplot
    plt.colorbar(im4, ax=axs[3], shrink=0.3, aspect=20)  # Colorbar for the first subplot
    im5 = axs[4].matshow(beta_5, cmap='PuOr', norm=norm)
    axs[4].axis('off')
    axs[4].set_title(captions[4], fontsize=12)  # Set caption for each subplot
    plt.colorbar(im5, ax=axs[4], shrink=0.3, aspect=20)  # Colorbar for the first subplot
    im6 = axs[5].matshow(beta_6, cmap='PuOr', norm=norm)
    axs[5].axis('off')
    axs[5].set_title(captions[5], fontsize=12)  # Set caption for each subplot
    plt.colorbar(im6, ax=axs[5], shrink=0.3, aspect=20)  # Colorbar for the first subplot
    # Adjust layout to leave space for colorbar
    fig.subplots_adjust(right=0.95)  # Leave space on the right for colorbar
    plt.show()

    # # simulated disease region map visualization
    # roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
    # roi_ds[0, idx_ds] = np.reshape(sigma2_p, shape=(n_v, 1))  # randomly select the 4-th subject
    # img_r = np.reshape(roi_ds, shape=(img_ds_h, img_ds_w))
    # # Display the array as an image
    # plt.imshow(img_r, cmap='viridis')
    # plt.colorbar()
    # plt.show()

    for i in range(20):
        roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
        roi_ds[0, idx_ds] = np.reshape(y_1[i, :], (n_v, 1))  # randomly select the 4-th subject
        img_r = np.reshape(roi_ds, (img_ds_h, img_ds_w))
        roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
        roi_ds[0, idx_ds] = np.reshape(y_1[i, :]+5 * bv[i, :], (n_v, 1))
        bv_ds = np.reshape(roi_ds, (img_ds_h, img_ds_w))
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
        captions = ['Thickness Map', 'True Disease Region', 'Detected Disease Region', 'ITE', 'TSE', 'ISE']  # Captions for each subplot
        fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(10, 5))
        im1 = axs[0].matshow(img_r, cmap='viridis')
        axs[0].axis('off')
        axs[0].set_title(captions[0], fontsize=12)  # Set caption for each subplot
        plt.colorbar(im1, ax=axs[0], shrink=0.3, aspect=20)  # Colorbar for the first subplot
        im2 = axs[1].matshow(bv_ds, cmap='viridis')
        axs[1].axis('off')
        axs[1].set_title(captions[1], fontsize=12)  # Set caption for each subplot
        plt.colorbar(im2, ax=axs[1], shrink=0.3, aspect=20)  # Colorbar for the first subplot
        im3 = axs[2].matshow(bv_hat, cmap='viridis')
        axs[2].axis('off')
        axs[2].set_title(captions[2], fontsize=12)  # Set caption for each subplot
        plt.colorbar(im3, ax=axs[2], shrink=0.3, aspect=20)  # Colorbar for the first subplot
        im4 = axs[3].matshow(ite, cmap='viridis')
        axs[3].axis('off')
        axs[3].set_title(captions[3], fontsize=12)  # Set caption for each subplot
        plt.colorbar(im4, ax=axs[3], shrink=0.3, aspect=20)  # Colorbar for the first subplot
        im5 = axs[4].matshow(tse, cmap='viridis')
        axs[4].axis('off')
        axs[4].set_title(captions[4], fontsize=12)  # Set caption for each subplot
        plt.colorbar(im5, ax=axs[4], shrink=0.3, aspect=20)  # Colorbar for the first subplot
        im6 = axs[5].matshow(ise, cmap='viridis')
        axs[5].axis('off')
        axs[5].set_title(captions[5], fontsize=12)  # Set caption for each subplot
        plt.colorbar(im6, ax=axs[5], shrink=0.3, aspect=20)  # Colorbar for the first subplot
        # Adjust layout to leave space for colorbar
        fig.subplots_adjust(right=0.95)  # Leave space on the right for colorbar
        plt.show()

if __name__ == "__main__":
    main()

