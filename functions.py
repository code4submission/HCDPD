""" all functions used in main command """

import numpy as np
# import pandas as pd
from sklearn.linear_model import LogisticRegression
# from scipy.stats import gamma, beta  #, bernoulli
from scipy.stats import truncnorm, norm
from scipy.ndimage import gaussian_filter
from numba import jit

""" Converts 2D coordinates to a 1D index in a NumPy array """
def coord_to_index_np(coords, shape):
    """
    Args:
        coords (tuple): The (x, y) coordinates.
        shape (tuple): The shape of the NumPy array (height, width).

    Returns:
        int: The corresponding 1D index.
    """
    return np.ravel_multi_index(coords, shape)

""" Implements soft thresholding """
def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

""" Update sigma2 """
def update_sigma2(res, a_0, b_0):
    n, n_v = res.shape
    sigma2_tt = np.zeros((n_v, 1))
    a = a_0 + n / 2
    for k in range(n_v):
        b = b_0 + 1 / 2 * np.sum(res[:, k] ** 2)
        sigma2_tt[k, 0] = 1 / np.random.gamma(a, scale=1 / b)
    return sigma2_tt

""" Update alpha """
def update_alpha(x, res, alpha_ini, alpha_n_ini, sigma2, lambda2_alpha, pi):

    # def normal_pdf(v, mean=0, std_dev=1):
    #     """ Compute the Probability Density Function (PDF) of a normal distribution """
    #     pi_math = 3.1415926
    #     coefficient = 1 / (std_dev * np.sqrt(2 * pi_math))
    #     exponent = np.exp(-((v - mean) ** 2) / (2 * std_dev ** 2))
    #     return coefficient * exponent

    n, p = x.shape
    n_v = res.shape[1]
    alpha = np.zeros((p, n_v))
    for j in range(p):
        for k in range(n_v):
            # Sample z_alpha_j,k
            var_j = sigma2[k, 0] / (np.sum(x[:, j]**2) + 1 / lambda2_alpha)
            mu_j = var_j * (alpha_n_ini[j, k]/(sigma2[k, 0]*lambda2_alpha) + np.sum(x[:, j] * (res[:, k] + x[:, j] * alpha_ini[j, k]))/sigma2[k, 0])
            posterior_slab = pi[j, k] * norm.pdf(0, loc=mu_j, scale=np.sqrt(var_j))
            posterior_spike = (1 - pi[j, k])
            # Sample alpha_j,k
            if posterior_slab >= posterior_spike or posterior_spike == 0:
                alpha[j, k] = np.random.normal(mu_j, np.sqrt(var_j))
            else:
                alpha[j, k] = 0
    return alpha

""" Update lambda2_alpha """
def update_lambda2_alpha(a_0, b_0, z_alpha, alpha_s):
    a = a_0 + 0.5*np.sum(z_alpha)
    b = b_0 + 0.5*np.sum(z_alpha*alpha_s)
    lambda2_alpha = 1 / np.random.gamma(a, scale=1 / b)
    return lambda2_alpha

""" Update nu_alpha """
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def update_nu_alpha(c_0, d_0, z_alpha):
    p, n_v = z_alpha.shape
    nu_alpha = 0*z_alpha
    for j in range(p):
        for k in range(n_v):
            c = c_0 + z_alpha[j, k]
            d = d_0 + 1 - z_alpha[j, k]
            nu_alpha[j, k] = np.random.beta(c, d)
    return nu_alpha

""" Update lambda2_beta """
def update_lambda2_beta(a_0, b_0, z_beta, beta_s):
    a = a_0 + 0.5*np.sum(z_beta)
    b = b_0 + 0.5*np.sum(z_beta*beta_s)
    lambda2_beta = 1 / np.random.gamma(a, scale=1 / b)
    return lambda2_beta

""" Update nu_beta """
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def update_nu_beta(c_0, d_0, z_beta):
    p, n_v = z_beta.shape
    nu_beta = 0*z_beta
    for j in range(p):
        for k in range(n_v):
            c = c_0 + z_beta[j, k]
            d = d_0 + 1 - z_beta[j, k]
            nu_beta[j, k] = np.random.beta(c, d)
    return nu_beta

""" Update gamma """
def update_gamma(res, b, sigma2, gamma_tt, sigma2_gamma):
    n, n_v = res.shape
    ga_all = []
    for k in range(n_v):
        if np.sum(b[:, k]) >= 5:
            var_ga = 1 / (b[:, k].T @ b[:, k] / sigma2[k, 0] + 1 / sigma2_gamma)
            mu_ga = var_ga * (gamma_tt / sigma2_gamma + b[:, k].T @ res[:, k] / sigma2[k, 0])
            # Sample gamma
            ga_tmp = truncnorm(-np.inf, -mu_ga/np.sqrt(var_ga), mu_ga, np.sqrt(var_ga))
            ga_sample = ga_tmp.rvs(1)
            ga_all.append(ga_sample)
    ga = np.median(ga_all)
    return ga

""" Update tau """
def update_tau(res, b_n, sigma2, tau_tt, sigma2_tau):
    n, n_v = res.shape
    tau_all = []
    for k in range(n_v):
        if np.sum(1*(b_n[:, k]>0)) >= 5:
            var_tau = 1 / (b_n[:, k].T @ b_n[:, k] / sigma2[k, 0] + 1 / sigma2_tau)
            mu_tau = var_tau * (tau_tt / sigma2_tau + b_n[:, k].T @ res[:, k] / sigma2[k, 0])
            # Sample tau
            tau_tmp = truncnorm(-np.inf, -mu_tau/np.sqrt(var_tau), mu_tau, np.sqrt(var_tau))
            tau_sample = tau_tmp.rvs(1)
            tau_all.append(tau_sample)
    tau = np.median(tau_all)
    return tau

""" Update beta, z_beta """
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def update_beta_fast(x, g_info, b, b_n, res_b0, res_b1, sigma2, omega_rv, lambda2_beta, beta_n_ini, pi):

    # def normal_pdf(v, mean=0, std_dev=1):
    #     """ Compute the Probability Density Function (PDF) of a normal distribution """
    #     pi_math = 3.1415926
    #     coefficient = 1 / (std_dev * np.sqrt(2 * pi_math))
    #     exponent = np.exp(-((v - mean) ** 2) / (2 * std_dev ** 2))
    #     return coefficient * exponent

    def reshape_contiguous_array(arr, new_shape):
        # Ensure the array is contiguous
        contig_arr = np.ascontiguousarray(arr)
        # Reshape the array
        reshaped_arr = contig_arr.reshape(new_shape)
        return reshaped_arr

    n, p = x.shape
    n_v = b.shape[1]
    beta_new = np.zeros((p + 2, n_v))
    b_new = np.zeros((n, n_v))
    # Fit logistic regression for each voxel
    for jj in range(n_v):
        b_jj = reshape_contiguous_array(b[:, jj], (n, 1))  # Response variable for this voxel
        if np.sum(b_jj) == 0:
            beta_new[:, jj] = 0
        else:
            b_jj_bar = reshape_contiguous_array(b_n[:, jj], (n, 1))
            x_jj = np.hstack((x, g_info, b_jj_bar))
            for kk in range(p+2):
                var_kk = 1 / (np.sum(omega_rv[:, kk] * x_jj[:, kk] ** 2) + 1 / (sigma2[jj, 0]*lambda2_beta))
                mu_kk = var_kk * (beta_n_ini[kk, jj] / (sigma2[jj, 0] * lambda2_beta) + np.sum(
                    x_jj[:, kk] * (b_jj - 0.5)))
                posterior_slab = pi[kk, jj] * norm.pdf(0, loc=mu_kk, scale=np.sqrt(var_kk))
                posterior_spike = (1 - pi[kk, jj])
                # Sample alpha_j,k
                if posterior_slab >= posterior_spike or posterior_spike == 0:
                    beta_new[kk, jj] = np.random.normal(mu_kk, np.sqrt(var_kk))
                else:
                    beta_new[kk, jj] = 0
            pi_b = 1 / (1 + np.exp(-np.dot(x_jj, beta_new[:, jj])))  # Probability of y=1
            pi_b_post = pi_b * np.exp(-0.5/sigma2[jj, 0]*res_b1[:, jj]**2) / (pi_b * np.exp(-0.5/sigma2[jj, 0]*res_b1[:, jj]**2) + (1-pi_b) * np.exp(-0.5/sigma2[jj, 0]*res_b0[:, jj]**2))
            b_new[:, jj] = 1 * (pi_b_post >=0.5)
    return beta_new, b_new

""" Update beta, z_beta """
def update_beta(x, g_info, b, b_n, res_b0, res_b1, sigma2, p_threshold):
    n, p = x.shape
    n_v = b.shape[1]
    beta_new = np.zeros((p + 2, n_v))
    b_new = np.zeros((n, n_v))
    # Fit logistic regression for each voxel
    for jj in range(n_v):
        b_jj = b[:, jj]  # Response variable for this voxel
        if np.sum(b_jj) == 0:
            beta_new[:, jj] = 0
        else:
            b_jj_bar = b_n[:, jj]
            x_jj = np.hstack((x, g_info, np.reshape(b_jj_bar, (n, 1))))
            # Logistic regression model
            model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
            #model.fit(ds[['Sex', 'BMI', 'Age', 'Sex_BMI', 'Sex_Age', 'B_jj_bar']], ds['B_jj'])
            model.fit(x_jj, b_jj)
            # Store coefficients
            beta_tmp = model.coef_.flatten()
            # print(beta_tmp)
            beta_new[:, jj] = soft_threshold(beta_tmp, 0.01)
            pi_b = model.predict_proba(x_jj)[:, 1]  # Probability of y=1
            pi_b_post = pi_b * np.exp(-0.5/sigma2[jj, 0]*res_b1[:, jj]**2) / (pi_b * np.exp(-0.5/sigma2[jj, 0]*res_b1[:, jj]**2) + (1-pi_b) * np.exp(-0.5/sigma2[jj, 0]*res_b0[:, jj]**2))
            b_new[:, jj] = 1 * (pi_b_post >=p_threshold)
    return beta_new, b_new

""" Update score_0 """
def update_score0(smoothed_res_0, pcs_ini):
    scores_0 = smoothed_res_0 @ pcs_ini
    return scores_0

""" Update score_1 """
def update_score1(res_1, img_ds_h, img_ds_w, idx_ds, pcs_ini):
    n, n_v = res_1.shape
    smoothed_res_1 = np.zeros(res_1.shape)
    for i in range(n):
        roi_ds = np.zeros(shape=(1, img_ds_h * img_ds_w))
        roi_ds[0, idx_ds] = np.reshape(res_1[i, :], (n_v, 1))
        img_2d = np.reshape(roi_ds, (img_ds_h, img_ds_w))
        # Apply Gaussian filter
        smoothed_img_2d = gaussian_filter(img_2d, sigma=1)
        smoothed_img_r = np.reshape(smoothed_img_2d, (1, img_ds_h * img_ds_w))
        smoothed_res_1[i, :] = np.reshape(smoothed_img_r[0, idx_ds], (n_v, ))
    scores_1 = smoothed_res_1 @ pcs_ini
    return scores_1
