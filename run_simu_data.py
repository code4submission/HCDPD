"""
##  Bayesian Causal Inference / Heterogeneous Causal
Framework for Disease Pattern Detection
##  OAI dataset
## Run simulation studies
## command line: python run_simu_data.py --id 1   (1 is the simulation id for random seed specification)
"""

import os
from os import mkdir
import argparse
import numpy as np
import random
# from scipy.stats import gamma #, bernoulli
from sklearn.decomposition import PCA
from functions import (soft_threshold, update_sigma2, update_alpha, update_beta, update_gamma, update_tau,
                       update_score0, update_score1)
from scipy.ndimage import gaussian_filter
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
import scipy.stats as stats
# from sklearn.linear_model import LogisticRegression
# from polyagamma import random_polyagamma
import time

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
    """ set random seed """
    random.seed(args.id)

    """ directory settings """
    real_data_path = './dat/real_data'
    simu_data_path = './dat/simu_data/simu_%s' % args.id
    simu_result_path = './dat/simu_result/simu_%s' % args.id
    if os.path.isdir(simu_result_path) is False:
        mkdir(simu_result_path)

    """ load real data """
    img_ds_h, img_ds_w = [62, 62]               # (down-sampled) image size 62*62
    idx_ds = np.load('%s/idx_ds.npy' % real_data_path)     # idx_ds. index (1D) for mask area (minus 1 s.t. initial with 0)
    n_v = idx_ds.shape[0]                           # number of pixels in mask area
    w_adj = np.load('%s/W.npy' % real_data_path)       # Adjacency matrix (n_v*n_v)
    n_v_neighbor = np.load('%s/n_v_neighbor.npy' % real_data_path)    # number of pixels in neighborhood (n_v*1)

    x_0 = np.load('%s/x_0.npy' % simu_data_path)   # covariates for nc: intercept, sex (F==1), bmi, age, sex_bmi, sex_age (n_0*6, n_0 = 200)
    y_0 = np.load('%s/y_0.npy' % simu_data_path)  # (down-sampled) thickness map (n*m) for nc (n_0*n_v)
    x_1 = np.load('%s/x_1.npy' % simu_data_path)   # covariates for OA patients: intercept, sex, bmi, age, sex_bmi, sex_age (n_0*6, n_0 = 200)
    y_1 = np.load('%s/y_1.npy' % simu_data_path)  # (down-sampled) thickness map (n*m) for oa (n_1*n_v)
    g_info = np.load('%s/g_info.npy' % simu_data_path)   # simulated KLG scores for oa
    n_0, p = x_0.shape
    n_1 = x_1.shape[0]

    print("Get initial estimation of alpha")
    # First smoothing, then estimation
    start_time = time.time()
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
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    print("Get initial estimation of pcs, scores_0, and sigma0")
    smoothed_res_ini = smoothed_y0 - x_0 @ alpha_ini
    # Perform PCA on the smoothed residuals
    pca = PCA()
    pca.fit(smoothed_res_ini)
    pc_coeff = pca.components_.T  # PCA components (transposed for column-wise similarity)
    pc_score = pca.fit_transform(smoothed_res_ini)
    # explained = pca.explained_variance_ratio_ * 100  # Explained variance percentage
    # cum_explained = np.cumsum(explained)  # Cumulative explained variance
    # print(cum_explained)
    # Select the top n_pcs principal components
    n_pcs = 3
    pcs = pc_coeff[:, :n_pcs]
    scores_0_ini = pc_score[:, :n_pcs]

    print("Get initial estimation of B")
    bv_ini = np.zeros(y_1.shape)
    res_1 = y_1 - x_1 @ alpha_ini
    for i in range(n_1):
        # We use a mask that limits to the foreground: the problem that we are
        # interested in here is not separating the objects from the background,
        # but separating them one from the other.
        roi_ds = np.zeros(shape=(1, img_ds_h*img_ds_w))
        roi_ds[0, idx_ds] = 1
        img_ds = np.reshape(roi_ds, (img_ds_h, img_ds_w))
        mask = img_ds.astype(bool)
        roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
        roi_ds[0, idx_ds] = np.reshape(res_1[i, :], (n_v, 1))
        img_r = np.reshape(roi_ds, (img_ds_h, img_ds_w))
        # Convert the image into a graph with the value of the gradient on the edges.
        graph = image.img_to_graph(img_r, mask=mask)
        # Take a decreasing function of the gradient: an exponential
        # The smaller bta is, the more independent the segmentation is of the
        # actual image. For bta=1, the segmentation is close to a voronoi
        bta = 1
        eps = 1e-6
        graph.data = np.exp(-bta * graph.data / graph.data.std()) + eps
        n_clusters = 50
        labels = spectral_clustering(graph, n_clusters=n_clusters, eigen_solver="arpack")
        label_im = np.full(mask.shape, -1.0)
        label_im[mask] = labels
        # Calculate average intensity for each cluster
        average_intensities = []
        for ii in range(n_clusters):
            mask_ii = (label_im == ii)
            average_intensities.append(np.mean(img_r[mask_ii]))
        average_intensities = np.array(average_intensities)
        # Find the region with the lowest intensity
        lowest_intensity_cluster = np.where(stats.zscore(average_intensities)<=-2.33)   # below 1% quantile
        if len(lowest_intensity_cluster[0]) == 0:
            lowest_intensity_cluster = np.where(average_intensities==np.min(average_intensities)) # the region with the lowest intensity
        # Highlight the region with the lowest intensity
        n_highlighted_region = len(lowest_intensity_cluster[0])
        highlighted_region = np.zeros(shape=(img_ds_h, img_ds_w))
        for kk in range(n_highlighted_region):
            highlighted_region = highlighted_region + (label_im == lowest_intensity_cluster[0][kk]) * 1
        highlighted_region_vec = np.reshape(highlighted_region, (1, img_ds_h * img_ds_w))
        bv_ini[i, :] = np.reshape(highlighted_region_vec[0, idx_ds], (n_v,))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    # Calculate B_n (scaled B using W and neighboring vertices)
    bv_n_ini = (bv_ini @ w_adj.T) / (np.ones((n_1, 1)) @ n_v_neighbor.T)

    """ set initial values for gamma and tau """
    gamma_ini = random.uniform(-0.4, -0.1)
    tau_ini = random.uniform(-0.4, -0.1)

    print("Set the initial values of score_1")
    res_1 = y_1 - x_1 @ alpha_ini - gamma_ini * bv_ini - tau_ini * bv_n_ini
    smoothed_res_1 = np.zeros(res_1.shape)
    for i in range(n_1):
        roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
        roi_ds[0, idx_ds] = np.reshape(res_1[i, :], (n_v, 1))
        img_2d = np.reshape(roi_ds, (img_ds_h, img_ds_w))
        # Apply Gaussian filter
        smoothed_img_2d = gaussian_filter(img_2d, sigma=1)
        smoothed_img_r = np.reshape(smoothed_img_2d, (1, img_ds_h * img_ds_w))
        smoothed_res_1[i, :] = np.reshape(smoothed_img_r[0, idx_ds], (n_v, ))
    scores_1_ini = smoothed_res_1 @ pcs
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    """ set hyperparameters """
    a_0 = 8
    b_0 = 10  # Equivalent of floor(1 / ((a_0 - 1) * sigma20_mean)), calculation omitted
    sigma2_gamma = 0.1
    sigma2_tau = 0.1
    p_threshold = 0.5

    """ MCMC iteration settings """
    n_iter = 550     # total iteration number
    # n_burnin = 200    # iteration number for burn-in stage
    # n_slice =  10     # slicing number
    sigma2_seq = np.zeros((n_iter, n_v))
    alpha_seq = np.zeros((n_iter, p, n_v))
    beta_seq = np.zeros((n_iter, p+2, n_v))
    # lambda2_alpha_seq = np.zeros((n_iter, 1))
    gamma_seq = np.zeros((n_iter, 1))
    tau_seq = np.zeros((n_iter, 1))
    bv_seq = np.zeros((n_iter, n_1, n_v))

    alpha_tt = alpha_ini
    scores_0_tt = scores_0_ini
    scores_1_tt = scores_1_ini
    gamma_tt = gamma_ini
    lambda2_alpha_tt = 0.1 # 1 / gamma.rvs(a_0, scale=1 / b_0)
    nu_alpha_tt = np.vstack((np.ones(n_v), 0.8 - 0.6 * (alpha_tt[1:p, :] == 0)))
    tau_tt = tau_ini
    bv_tt = bv_ini
    bv_n_tt = (bv_tt @ w_adj.T) / (np.ones((n_1, 1)) @ n_v_neighbor.T)

    print("Gibbs sampling starts...")
    for tt in range(n_iter):
        if np.remainder(tt+1, 100) == 0:
            print(f'iteration {tt+1}')
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time} seconds")
        # update sigma20
        res_tt = y_0 - x_0 @ alpha_tt - scores_0_tt @ pcs.T
        sigma2_tt = update_sigma2(res_tt, a_0, b_0)
        sigma2_seq[tt, :] = np.reshape(sigma2_tt, (n_v, ))

        # update alpha & z_alpha
        alpha_n_tt = (alpha_tt @ w_adj.T) / (np.ones((p, 1)) @ n_v_neighbor.T)
        alpha_tt = update_alpha(x_0, res_tt, alpha_tt, alpha_n_tt, sigma2_tt, lambda2_alpha_tt, nu_alpha_tt)
        alpha_seq[tt, :, :] = alpha_tt

        # update gamma
        res_1_tt = y_1 - x_1 @ alpha_tt - tau_tt * bv_n_tt - scores_1_tt @ pcs.T
        gamma_tt = update_gamma(res_1_tt, bv_tt, sigma2_tt, gamma_tt, sigma2_gamma)
        gamma_seq[tt, 0] = gamma_tt

        # update tau
        res_2_tt = y_1 - x_1 @ alpha_tt - gamma_tt * bv_tt - scores_1_tt @ pcs.T
        tau_tt = update_tau(res_2_tt, bv_n_tt, sigma2_tt, tau_tt, sigma2_tau)
        tau_seq[tt, 0] = tau_tt

        # update beta, b, and b_n
        res_3_b0 = y_1 - x_1 @ alpha_tt - tau_tt * bv_n_tt - scores_1_tt @ pcs.T
        res_3_b1 = y_1 - x_1 @ alpha_tt - gamma_tt * bv_tt - tau_tt * bv_n_tt - scores_1_tt @ pcs.T
        beta_tt, bv_tt = update_beta(x_1, g_info, bv_tt, bv_n_tt, res_3_b0, res_3_b1, sigma2_tt, p_threshold)
        bv_n_tt = (bv_tt @ w_adj.T) / (np.ones((n_1, 1)) @ n_v_neighbor.T)
        beta_seq[tt, :, :] = beta_tt
        bv_seq[tt, :, :] = bv_tt

        # update score_0
        res_0 = smoothed_y0 - x_0 @ alpha_tt
        scores_0_tt = update_score0(res_0, pcs)

        # update score_1
        res_1 = y_1 - x_1 @ alpha_tt - gamma_tt * bv_tt - tau_tt * bv_n_tt
        scores_1_tt = update_score1(res_1, img_ds_h, img_ds_w, idx_ds, pcs)

    # save all the MCMC samples
    np.save('%s/sigma2_seq.npy' % simu_result_path, sigma2_seq)
    np.save('%s/alpha_seq.npy' % simu_result_path, alpha_seq)
    np.save('%s/beta_seq.npy' % simu_result_path, beta_seq)
    np.save('%s/gamma_seq.npy' % simu_result_path, gamma_seq)
    np.save('%s/tau_seq.npy' % simu_result_path, tau_seq)
    np.save('%s/bv_seq.npy' % simu_result_path, bv_seq)
    np.save('%s/pcs.npy' % simu_result_path, pcs)


if __name__ == "__main__":
    main()