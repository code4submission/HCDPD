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
from sklearn.metrics import (rand_score, adjusted_rand_score, normalized_mutual_info_score, homogeneity_score,
                             completeness_score, accuracy_score)
import seaborn as sns
# from sklearn.decomposition import PCA
# from functions import soft_threshold, update_sigma2, update_alpha, update_beta, update_gamma, update_tau, update_score1
# from scipy.ndimage import gaussian_filter
# from sklearn.feature_extraction import image
# from sklearn.cluster import spectral_clustering
# import scipy.stats as stats
# from sklearn.linear_model import LogisticRegression
import pandas as pd
import argparse

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
    parser = argparse.ArgumentParser(
        description="Run Causal Inference / Abnormal Region Detection on Simulation Datasets.")
    parser.add_argument("--p", type=float, required=True, help="threshold value")
    args = parser.parse_args()

    p_threshold = float(args.p)
    """ directory settings """
    real_data_path = './dat/real_data'

    n_simu = 60  # number of simulation datasets

    # MCMC settings
    n_iter = 550  # total iteration number
    n_burnin = 50  # iteration number for burn-in stage
    n_slice = 10  # slicing number

    # img_ds_h, img_ds_w = [62, 62]  # (down-sampled) image size 62*62
    # idx_ds = np.load('%s/idx_ds.npy' % real_data_path)  # idx_ds. index (1D) for mask area (minus 1 s.t. initial with 0)
    # n_v = idx_ds.shape[0]  # number of pixels in mask area
    w_adj = np.load('%s/W.npy' % real_data_path)  # Adjacency matrix (n_v*n_v)
    n_v_neighbor = np.load('%s/n_v_neighbor.npy' % real_data_path)  # number of pixels in neighborhood (n_v*1)

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

    n_1 = 180
    b_ri_all = np.zeros(shape=(n_1, n_simu))
    b_ari_all = np.zeros(shape=(n_1, n_simu))
    b_homo_all = np.zeros(shape=(n_1, n_simu))
    b_nmi_all = np.zeros(shape=(n_1, n_simu))
    b_comp_all = np.zeros(shape=(n_1, n_simu))
    alpha_mse = np.zeros(shape=(n_simu, 1))
    alpha_acc = np.zeros(shape=(n_simu, 1))
    beta_mse = np.zeros(shape=(n_simu, 1))
    beta_acc = np.zeros(shape=(n_simu, 1))
    gamma_hat = np.zeros(shape=(n_simu, 1))
    tse_mse = np.zeros(shape=(n_simu, 1))
    ise_mse = np.zeros(shape=(n_simu, 1))
    dse_mse = np.zeros(shape=(n_simu, 1))

    gamma0 = -0.15
    tau0 = -0.2

    for j in range(n_simu):
        simu_id = j + 1
        simu_data_path = './dat/simu_data/simu_%s' % simu_id
        simu_result_path = './dat/simu_result/simu_%s' % simu_id
        bv0 = np.load('%s/bv.npy' % simu_data_path)
        bv0_n = (bv0 @ w_adj.T) / (np.ones((n_1, 1)) @ n_v_neighbor.T)
        # y_1 = np.load('%s/y_1.npy' % simu_data_path)
        # x_1 = np.load('%s/x_1.npy' % simu_data_path)
        bv_seq = np.load('%s/bv_seq.npy' % simu_result_path)
        bv_p = 1 * (np.mean(bv_seq[n_burnin:n_iter:n_slice, :, :], axis=0) >= p_threshold)
        for i in range(n_1):
            b_ri_all[i, j] = rand_score(bv0[i, :], bv_p[i, :])
            b_ari_all[i, j] = adjusted_rand_score(bv0[i, :], bv_p[i, :])
            b_homo_all[i, j] = homogeneity_score(bv0[i, :], bv_p[i, :])
            b_nmi_all[i, j] = normalized_mutual_info_score(bv0[i, :], bv_p[i, :])
            b_comp_all[i, j] = completeness_score(bv0[i, :], bv_p[i, :])
            bv_n_p = (bv_p @ w_adj.T) / (np.ones((n_1, 1)) @ n_v_neighbor.T)
            alpha_seq = np.load('%s/alpha_seq.npy' % simu_result_path)
            alpha_p = np.median(alpha_seq[n_burnin:n_iter:n_slice, :, :], axis=0)
            z_alpha = 1 * (np.abs(alpha_p) <= 0.01)
            alpha0 = np.load('%s/alpha0.npy' % simu_data_path)
            z_alpha0 = 1 * (alpha0 == 0)
            alpha_acc[j, 0] = accuracy_score(z_alpha.flatten(), z_alpha0.flatten())
            alpha_mse[j, 0] = np.median((alpha_p[np.where(alpha0 != 0)] /alpha0[np.where(alpha0 != 0)] - 1)  ** 2)
            beta_seq = np.load('%s/beta_seq.npy' % simu_result_path)
            beta_p = np.median(beta_seq[n_burnin:n_iter:n_slice, :, :], axis=0)
            z_beta = 1 * (np.abs(beta_p) <= 0.01)
            beta0 = np.load('%s/beta0.npy' % simu_data_path)
            z_beta0 = 1 * (beta0 == 0)
            beta_acc[j, 0] = accuracy_score(z_beta.flatten(), z_beta0.flatten())
            beta_mse[j, 0] = np.median((beta_p[np.where(beta0 != 0)] / beta0[np.where(beta0 != 0)] - 1) ** 2)
            gamma_seq = np.load('%s/gamma_seq.npy' % simu_result_path)
            gamma_p = np.mean(gamma_seq[n_burnin:n_iter:n_slice])
            gamma_hat[j, 0] = gamma_p
            tau_seq = np.load('%s/tau_seq.npy' % simu_result_path)
            tau_p = np.mean(tau_seq[n_burnin:n_iter:n_slice])
            dse_mse[j, 0] = (gamma_p - gamma0)**2
            tse_mse[j, 0] = np.mean((gamma_p + tau_p * bv_n_p - gamma0 - tau0 * bv0_n) ** 2)
            ise_mse[j, 0] = np.mean((tau_p * bv_n_p - tau0 * bv0_n) ** 2)
    b_ri = np.median(b_ri_all, axis=0)
    b_ari = np.median(b_ari_all, axis=0)
    b_homo = np.median(b_homo_all, axis=0)
    b_nmi = np.median(b_nmi_all, axis=0)
    b_comp = np.median(b_comp_all, axis=0)
    # b_v_measure = np.median(b_v_measure_all, axis=0)

    print(np.mean(dse_mse))

    # disease region detection accuracy visualization
    sns.set_theme(font_scale=1.5)  # Adjust the value as needed

    sns.set_theme(style="ticks")
    dat = {
        'Values': np.concatenate([b_ri.ravel(),
                                  b_ari.ravel(),
                                  b_homo.ravel(),
                                  b_comp.ravel(),
                                  #b_v_measure.ravel(),
                                  b_nmi.ravel()]),
        'Group': [r'RI'] * n_simu + [r'ARI'] * n_simu
                 + [r'HOM'] * n_simu
                 + [r'COM'] * n_simu
                 # + [r'V-Measure'] * n_simu
                 + [r'NMI'] * n_simu
    }
    df = pd.DataFrame(dat)
    # Create Seaborn boxplot
    sns.boxplot(x='Group', y='Values', data=df)
    # Add in points to show each observation
    sns.stripplot(data=df, x="Group", y="Values", size=4, color=".3")
    # Customizations
    plt.title('Disease Region Detection Performance')
    plt.xlabel('Metric')
    plt.ylabel('')
    # Show plot
    plt.show()

    # estimation accuracy visualization
    sns.set_theme(font_scale=1.5)  # Adjust the value as needed

    sns.set_theme(style="ticks")
    dat = {
        'Values': np.concatenate([alpha_mse.ravel(), alpha_acc.ravel(), beta_mse.ravel(), beta_acc.ravel()]),
        'Group': [r'MSE: $\alpha_s$'] * n_simu + [r'ACC: $\alpha_s$'] * n_simu
                 + [r'MSE: $\beta_s, \varphi_s$'] * n_simu + [r'ACC: $\beta_s, \varphi_s$'] * n_simu
    }
    df = pd.DataFrame(dat)
    # Create Seaborn boxplot
    sns.boxplot(x='Group', y='Values', data=df)
    # Add in points to show each observation
    sns.stripplot(data=df, x="Group", y="Values", size=4, color=".3")
    # Customizations
    plt.title('Estimation Performance of Varying Coefficients')
    plt.xlabel('')
    plt.ylabel('')
    # Show plot
    plt.show()

    # estimation accuracy visualization
    sns.set_theme(font_scale=1.5)  # Adjust the value as needed

    sns.set_theme(style="ticks")
    dat = {
        'Values': np.concatenate([tse_mse.ravel(),
                                  ise_mse.ravel(),
                                  # dse_mse.ravel()
                                  ]),
        'Group': [r'TSE'] * n_simu + [r'ISE'] * n_simu  # + [r'DSE'] * n_simu
    }
    df = pd.DataFrame(dat)
    # Create Seaborn boxplot
    sns.boxplot(x='Group', y='Values', data=df)
    # Add in points to show each observation
    sns.stripplot(data=df, x="Group", y="Values", size=4, color=".3")
    # Customizations
    plt.title('Estimation Performance of Causal Estimands')
    plt.xlabel('')
    plt.ylabel('')
    # Show plot
    plt.show()


if __name__ == "__main__":
    main()
