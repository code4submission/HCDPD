"""
##  Bayesian Causal Inference
## Heterogeneous Causal Framework for Disease Pattern Detection
##  OAI dataset
## Generate simulation dataset
## command line: python gen_simu_data.py --id 1   (1 is the simulation id for random seed specification)
"""

import os
import argparse
import random
from os import mkdir
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from functions import soft_threshold

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

    print('Generate simulation dataset %s.' % args.id)
    """ set random seed """
    random.seed(args.id)

    """ directory settings """
    real_data_path = './dat/real_data'
    simu_data_path = './dat/simu_data/simu_%s' % args.id
    if os.path.isdir(simu_data_path) is False:
        mkdir(simu_data_path)

    """ load real data """
    dx = np.load('%s/dx.npy' % real_data_path)   # dx. diagnosis information: normal control (0) and patient (>0)
    x = np.load('%s/x.npy' % real_data_path)  # x. covariates: sex (F==1), bmi, age
    y_ds = np.load('%s/y_ds.npy' % real_data_path)   # y_ds. (down-sampled) thickness map (n*m)
    img_ds_h, img_ds_w = [62, 62]               # (down-sampled) image size 62*62
    idx_ds = np.load('%s/idx_ds.npy' % real_data_path)     # idx_ds. index (1D) for mask area (minus 1 s.t. initial with 0)
    n_v = idx_ds.shape[0]                           # number of pixels in mask area

    """ construct the neighbor system """
    # Initialize the landmarks array
    landmarks = np.zeros((n_v, 2))
    # Unravel the indices and assign them to landmarks
    indices = np.unravel_index(np.ravel(idx_ds), (img_ds_h, img_ds_w))
    landmarks[:, 0] = indices[0].flatten()  # Assign first component
    landmarks[:, 1] = indices[1].flatten()  # Assign second component
    # Initialize neighbor dictionary and adjacency matrix
    neighbor_dict = [None] * n_v
    w_adj = np.zeros((n_v, n_v))  # Adjacency matrix
    h = np.sqrt(2)  # Size of neighborhood (tunable)
    n_v_neighbor = np.zeros((n_v, 1), dtype=int)
    # Compute neighbors and populate adjacency matrix
    for kk in range(n_v):
        dist_k = cdist(landmarks, landmarks[kk, :].reshape(1, -1))[:, 0]  # Pairwise distances
        neighbors = np.where((dist_k > 0) & (dist_k <= h))[0]  # Find neighbors
        neighbor_dict[kk] = neighbors
        w_adj[kk, neighbors] = 1
        n_v_neighbor[kk, 0] = len(neighbors)
    np.save('%s/W.npy' % real_data_path, w_adj)
    np.save('%s/n_v_neighbor.npy' % real_data_path, n_v_neighbor)

    """ simulation settings """
    n_0 = 200
    y_nc = y_ds[dx[:, 0] == 0, :]  # Filter rows where dx == 0
    y0 = y_nc[:n_0, :]        # Take the first n_0 rows

    # Filter rows where dx == 0
    x_nc = x[dx[:, 0] == 0, :]  # Equivalent to x_nc = x(dx==0, :)

    # Extract and process columns
    sex = x_nc[:n_0, 0]                           # First column (sex)
    bmi = zscore(x_nc[:n_0, 1])                   # Z-score normalization for BMI (second column)
    age = zscore(x_nc[:n_0, 2])                   # Z-score normalization for age (third column)
    sex_bmi = sex * bmi             # Z-score normalization for sex * age
    sex_age = sex * age             # Z-score normalization for sex * age

    # Construct the design matrix
    x_0 = np.column_stack((np.ones(n_0), sex, bmi, age, sex_bmi, sex_age))
    # Save X_0
    np.save('%s/x_0.npy' % simu_data_path, x_0)
    # First smoothing, then estimation
    smoothed_y0 = y0
    for i in range(n_0):
        roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
        roi_ds[0, idx_ds] = np.reshape(y0[i, :], (n_v, 1))
        img_2d = np.reshape(roi_ds, (img_ds_h, img_ds_w))
        # Apply Gaussian filter
        smoothed_img_2d = gaussian_filter(img_2d, sigma=1)
        smoothed_img_r = np.reshape(smoothed_img_2d, (1, img_ds_h * img_ds_w))
        smoothed_y0[i, :] = np.reshape(smoothed_img_r[0, idx_ds], (n_v,))
    alpha0 = soft_threshold(np.linalg.solve(x_0.T @ x_0, x_0.T @ smoothed_y0), 0.01)

    # Save alpha0
    alpha0_path = '%s/alpha0.npy' % simu_data_path
    np.save(alpha0_path, alpha0)
    # Calculate z_alpha
    z_alpha = 1 - 1 * (alpha0[1:, :] == 0)
    # Save z_alpha
    z_alpha_path = '%s/z_alpha.npy' % simu_data_path
    np.save(z_alpha_path, z_alpha)

    # Perform PCA on the residuals
    smoothed_res0 = smoothed_y0 - x_0 @ alpha0
    pca = PCA()
    pca.fit(smoothed_res0)
    pc_coeff = pca.components_.T  # PCA components (transposed for column-wise similarity)
    # Select the top n_pcs principal components
    n_pcs = 3
    pcs0 = pc_coeff[:, :n_pcs]
    # Save pcs0
    np.save('%s/pcs0.npy' % simu_data_path, pcs0)
    scores_0 = np.random.randn(n_0, n_pcs) * (np.ones((n_0, 1)) @ np.reshape(np.arange(n_pcs*2, 0, (1-n_pcs*2)/2), (1, n_pcs)))
    np.save('%s/Scores_0.npy' % simu_data_path, scores_0)
    # Calculate residuals and sigma20
    sigma0 = 0.1  # np.var(eps0, axis=0, ddof=0)
    np.save('%s/sigma0.npy' % simu_data_path, sigma0)

    """ Thickness map generation (normal controls) """
    res_0 = np.random.randn(n_0, n_v) * sigma0
    y_0 = x_0 @ alpha0 + scores_0 @ pcs0.T + res_0
    # Save the thickness maps for normal controls (g=0)
    np.save('%s/y_0.npy' % simu_data_path, y_0)

    """ Set the true value for B_i """
    n_1 = 120  # Sample size for OA patients
    pt_1 = np.array([20, 16])  # Trigger point 1
    pt_2 = np.array([50, 20])  # Trigger point 2
    pt_3 = np.array([45, 30])  # Trigger point 3
    pt_4 = np.array([15, 50])  # Trigger point 4
    pt_5 = np.array([25, 48])  # Trigger point 5
    pt_6 = np.array([35, 42])  # Trigger point 6
    pts = np.vstack((pt_1, pt_2, pt_3, pt_4, pt_5, pt_6))
    pt_ids = [0, 1, 2, 3, 4, 5]
    # Initialize arrays
    bv = np.zeros((n_1, n_v))  # Matrix of zeros with size n_1 x n_v
    g_info = np.zeros((n_1, 1))  # Column vector of zeros with length n_1
    # Iterate over each subject
    for i in range(n_1):
        u = np.random.randint(1, 5)  # Random integer between 1 and 4
        if u == 1:
            g_info[i, 0] = 1
            pts_id = random.sample(pt_ids, 1)
            pts_1 = pts[pts_id, :] + [np.random.randint(-2, 3), np.random.randint(-2, 3)]
            dist_i = cdist(landmarks, pts_1.reshape(1, -1))[:, 0]  # Compute pairwise distance
            h = np.random.randint(2, 4)
            neighbor_dict_i = np.where(dist_i.flatten() <= h)[0]
            bv[i, neighbor_dict_i] = 1
            # print(f'Diseased region size is {len(neighbor_dict_i)} for the {i+1}-th subject')

        elif u == 2:
            g_info[i, 0] = 2
            pts_id = random.sample(pt_ids, 1)
            pts_1 = pts[pts_id, :] + [np.random.randint(-2, 3), np.random.randint(-2, 3)]
            dist_i = cdist(landmarks, pts_1.reshape(1, -1))[:, 0]  # Compute pairwise distance
            h = np.random.randint(4, 6)
            neighbor_dict_i = np.where(dist_i.flatten() <= h)[0]
            bv[i, neighbor_dict_i] = 1
            # print(f'Diseased region size is {len(neighbor_dict_i)} for the {i+1}-th subject')

        elif u >= 3:
            g_info[i, 0] = 3
            pts_id = random.sample(pt_ids, 2)
            pts_1 = pts[pts_id[0], :] + [np.random.randint(-2, 3), np.random.randint(-2, 3)]
            dist_i = cdist(landmarks, pts_1.reshape(1, -1))[:, 0]  # Compute pairwise distance
            h = np.random.randint(4, 6)
            neighbor_dict_i = np.where(dist_i.flatten() <= h)[0]
            bv[i, neighbor_dict_i] = 1
            # print(f'Diseased region size is {len(neighbor_dict_i)} for the {i+1}-th subject')
            pts_2 = pts[pts_id[1], :] + [np.random.randint(-2, 3), np.random.randint(-2, 3)]
            dist_i = cdist(landmarks, pts_2.reshape(1, -1))[:, 0]  # Compute pairwise distance
            h = np.random.randint(4, 6)
            neighbor_dict_i = np.where(dist_i.flatten() <= h)[0]
            bv[i, neighbor_dict_i] = 1
            # print(f'Diseased region size is {len(neighbor_dict_i)} for the {i+1}-th subject')

    # Save bv matrix
    np.save('%s/bv.npy' % simu_data_path, bv)
    # Save x_1
    np.save('%s/g_info.npy' % simu_data_path, g_info)

    """ Set the true value for beta_s """
    # Filter rows where dx >= 3
    x_oa = x[dx[:, 0] >= 3, :]
    # Extract and process predictors
    sex = x_oa[:n_1, 0]                           # First column (sex)
    bmi = zscore(x_oa[:n_1, 1])                   # Z-score normalization for BMI (second column)
    age = zscore(x_oa[:n_1, 2])                   # Z-score normalization for age (third column)
    sex_bmi = sex * bmi             # Z-score normalization for sex * age
    sex_age = sex * age             # Z-score normalization for sex * age
    # Construct the design matrix
    x_1 = np.column_stack((np.ones(n_1), sex, bmi, age, sex_bmi, sex_age))
    # Save x_1
    np.save('%s/x_1.npy' % simu_data_path, x_1)

    # Initialize beta_tmp
    beta_tmp = np.zeros((x_1.shape[1]+2, bv.shape[1]))

    # Fit logistic regression for each voxel
    for jj in range(bv.shape[1]):
        b_jj = bv[:, jj]  # Response variable for this voxel
        if np.sum(b_jj) == 0:
            beta_tmp[:, jj] = 0
        else:
            b_jj_bar = np.mean(bv[:, neighbor_dict[jj]], axis=1)
            x_jj = np.hstack((x_1, g_info, np.reshape(b_jj_bar, (n_1, 1))))
            # Logistic regression model
            model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
            # model.fit(ds[['Sex', 'BMI', 'Age', 'Sex_BMI', 'Sex_Age', 'B_jj_bar']], ds['B_jj'])
            model.fit(x_jj, b_jj)
            # Store coefficients
            beta_tmp[:, jj]  = model.coef_.flatten()

    # Assuming beta_tmp is already defined as a NumPy array
    beta0 = soft_threshold(beta_tmp, 0.01)
    # Save beta0
    beta0_path = '%s/beta0.npy' % simu_data_path
    np.save(beta0_path, beta0)
    # Calculate z_alpha
    z_beta = 1 - 1 * (beta0[1:, :] == 0)
    # Save z_beta
    z_beta_path = '%s/z_beta.npy' % simu_data_path
    np.save(z_beta_path, z_beta)

    """ Set the true values for gamma and tau """
    gamma0 = -0.2
    tau0 = -0.2
    # Save gamma0 and tau0
    gamma0_path = '%s/gamma0.npy' % simu_data_path
    np.save(gamma0_path, gamma0)
    tau0_path = '%s/tau0.npy' % simu_data_path
    np.save(tau0_path, tau0)

    """ Extract subsets of data for OA patients """
    # Calculate bv_n (scaled bv using w_adj and neighboring vertices)
    bv_n = (bv @ w_adj.T) / (np.ones((n_1, 1)) @ n_v_neighbor.T)
    # Compute scores_1
    scores_1 = np.random.randn(n_1, n_pcs) * (np.ones((n_1, 1)) @ np.reshape(np.arange(n_pcs*2, 0, (1-n_pcs*2)/2), (1, n_pcs)))
    # Save scores_1
    np.save('%s/scores_1.npy' % simu_data_path, scores_1)
    # Generate residuals res_1
    res_1 = np.random.randn(n_1, n_v) * sigma0
    # Generate Y_1
    y_1 = x_1 @ alpha0 + gamma0 * bv + tau0 * bv_n + scores_1 @ pcs0.T + res_1
    # Save Y_1
    y_1_path = '%s/y_1.npy' % simu_data_path
    np.save(y_1_path, y_1)

    print('Done.')

if __name__ == "__main__":
    main()