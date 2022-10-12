from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(1, '../general/')
from plot_general import hist_1d_2d

Y_model = np.loadtxt('../../data/running_coupling/output')

SS  =  StandardScaler(copy=True)
Npc = 4
pca = PCA(copy=True, whiten=True, svd_solver='full')
# Keep only the first `npc` principal components
pc_tf_data = pca.fit_transform(SS.fit_transform(Y_model)) [:,:Npc]
np.savetxt('../../data/running_coupling/PCA_transformed_data', pc_tf_data)

i, j = 2, 3
X = pc_tf_data[:,i]
Y = pc_tf_data[:,j]
hist_1d_2d(X, Y, 'dp%d'%i, 'dp%d'%j)
plt.savefig("../../plots/running_coupling/check_transformed_corr_%d%d"%(i, j))

# The transformation matrix from PC to Physical space
inverse_tf_matrix = pca.components_ * np.sqrt(pca.explained_variance_[:, np.newaxis]) * SS.scale_ 
inverse_tf_matrix = inverse_tf_matrix[:Npc,:]

np.savetxt('../../data/running_coupling/PCA_transform_matrix', inverse_tf_matrix)

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(7,2.6))
#importance = pca_analysis.explained_variance_
importance = pca.explained_variance_
cumulateive_importance = np.cumsum(importance)/np.sum(importance)
idx = np.arange(1,1+len(importance))
ax1.bar(idx,importance)
ax1.set_xlabel("PC index")
ax1.set_ylabel("Variance")
ax2.bar(idx,cumulateive_importance)
ax2.set_xlabel(r"The first $n$ PC")
ax2.set_ylabel("Fraction of total variance")
plt.tight_layout()
plt.savefig("../../plots/running_coupling/PC_importance")
