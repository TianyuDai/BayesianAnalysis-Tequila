from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

Y_model = np.loadtxt('../../data/dp_output/output')

SS  =  StandardScaler(copy=True)
Npc = 3
pca = PCA(copy=True, whiten=True, svd_solver='full')
# Keep only the first `npc` principal components
pc_tf_data = pca.fit_transform(SS.fit_transform(Y_model)) [:,:Npc]
np.savetxt('../../data/PCA_transformed_data', pc_tf_data)

# The transformation matrix from PC to Physical space
inverse_tf_matrix = pca.components_ * np.sqrt(pca.explained_variance_[:, np.newaxis]) * SS.scale_ 
inverse_tf_matrix = inverse_tf_matrix[:Npc,:]

np.savetxt('../../data/PCA_transform_matrix', inverse_tf_matrix)

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
plt.savefig("../../plots/PC_importance")
