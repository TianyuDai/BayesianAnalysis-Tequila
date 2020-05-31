import subprocess
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process import kernels
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from scipy import stats
import emcee
import numpy as np
# model data
data

npc = 3
scaler = StandardScaler(copy=True)
pca = PCA(copy=True, whiten=True, svd_solver='full')

# Keep only the first `npc` principal components
Z = pca.fit_transform(scaler.fit_transform(model_data))[:,:npc]

# The transformation matrix from PC to Physical space
Trans_Matrix = (  pca.components_
                * np.sqrt(pca.explained_variance_[:, np.newaxis])
                * scaler.scale_)
# Estimate the covariance of the negelected PCs
Residual_Cov = np.dot(Trans_Matrix[npc:].T, Trans_Matrix[npc:])

