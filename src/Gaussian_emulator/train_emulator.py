import time, sys, os
import pickle
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process import kernels as krnl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
sys.path.insert(1, '../general/')
from data_general import data_path
import matplotlib.pyplot as plt

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

Npc = 3
design_ptp = np.array([1-0.05, 0.5-0.05, 0.5-0.05])
design = np.loadtxt('../../data/lhd_sampling_3d_temp.txt')
pc_tf_data = np.loadtxt('../../data/PCA_transformed_data')

overide = False
EMU = "../../data/emulators.dat"
if (os.path.exists(EMU)) and (overide==False):
    print('Saved emulators exists and overide is prohibited')
    with open(EMU,"rb") as f:
        Emulators=pickle.load(f)
else:
    Emulators=[]
    for i in range(0,Npc):
        start_time = time.time()
        kernel=1*krnl.RBF(length_scale=design_ptp,
                          length_scale_bounds=np.outer(design_ptp, (1e-2, 1e2))) + krnl.WhiteKernel(noise_level=.1, 
                                  noise_level_bounds=(1e-2, 1e2))
        print("-----------------")
        print("Training PC #",i+1)
        GPR=gpr(kernel=kernel,n_restarts_optimizer=5)
        GPR.fit(design, pc_tf_data[:,i].reshape(-1,1))
        print('GPR score: {:1.3f}'.format(GPR.score(design,pc_tf_data[:,i])) )
        print("time: {:1.3f} seconds".format(time.time() - start_time))
        Emulators.append(GPR)

if (overide==True) or not (os.path.exists(EMU)):
    with open(EMU, "wb") as f:
        pickle.dump(Emulators,f)

Xdim = 3
use_NL = False
def predict_observables(model_parameters, diag_std=False):
    """Predicts the observables for any model parameter value using the trained emulators.
    
    Parameters
    ----------
    Theta_input : Model parameter values.
    Return
    ------
    Mean value and full error covaraiance matrix of the prediction is returened. """
    
    mean=[]
    variance=[]
    theta=np.array(model_parameters).flatten()
    print(theta)
    if len(theta)!=Xdim:
        raise TypeError('The input model_parameters array does not have the right dimensions')
    else: 
        theta=np.array(theta).reshape(1,Xdim)
        for i in range(Npc):
            mn,std=Emulators[i].predict(theta, return_std=True)
            mean.append(mn)
            variance.append(std**2)
    mean=np.array(mean).reshape(1,-1)
    inverse_transformed_mean = mean@inverse_tf_matrix + np.array(SS.mean_).reshape(1,-1)    
    variance_matrix = np.diag(np.array(variance).flatten())
    inverse_transformed_variance = np.einsum('ik,kl,lj-> ij', inverse_tf_matrix.T, variance_matrix, inverse_tf_matrix, 
                                             optimize=False)
    if use_NL:
        inverse_transformed_mean = inverse_transformed_mean**2
        inverse_transformed_variance *= np.outer(2.*inverse_transformed_mean[0]**.5, 
                                                 2.*inverse_transformed_mean[0]**.5)
    if diag_std:
        return inverse_transformed_mean[0], np.sqrt(np.diag(inverse_transformed_variance))
    else:
        return inverse_transformed_mean[0], inverse_transformed_variance

design_min = np.array([0.05, 0.05, 0.05])
inverse_tf_matrix = np.loadtxt('../../data/PCA_transform_matrix')

# randomly sample another 2 parametr points from the parameter space
np.random.seed(9)
# X_validation = design_min + np.random.rand(2, Xdim)*design_ptp
X_validation = np.array([[0.596, 1.51, 3.99]])

# Next, get the emulator prediction and uncertainty
A = np.array([predict_observables(it, diag_std=True) for it in X_validation])
Y_predicted = A[:,0,:]
Y_std = A[:,1,:]
np.savetxt('../../data/data_predicted', Y_predicted)
"""
# Model calculation at these two points
# Y_validation = np.array([np.concatenate(ToyModel(param)) for param in X_validation])
Y_validation = np.loadtxt('/home/td115/research/Result/JETSCAPE3.0/Tequila_test/AuAu200/centrality_0-10/AA200_hadrons_cross_section_pT.txt')

# plot the prediction + uncertainty band with the true model caluclation
fig, axes = plt.subplots(1,2, figsize=(7,3.5), sharex=True)
for i, (mean, std) in enumerate(zip(Y_predicted, Y_std)):
    label = 'GP emulated' if i==0 else''
    axes[0].fill_between(cen,mean[:Nc]-std[:Nc],mean[:Nc]+std[:Nc],color='r',alpha=.5, label=label)
    axes[1].fill_between(cen,mean[Nc:]-std[Nc:],mean[Nc:]+std[Nc:],color='r',alpha=.5, label=label)
for i, ym in enumerate(Y_validation):
    label = 'Model calc.' if i==0 else''
    axes[0].plot(cen, ym[:Nc], 'b.', label=label)
    axes[1].plot(cen, ym[Nc:], 'b.', label=label)

# Add labels
labels = r"$dN/d\eta$", r"$v_2$", r"$", r"$dN/d\eta$: pred-true", r"$v_2$: pred-true"
for ax, label in zip(axes, labels):
    ax.set_xlabel("Centrality(%)")
    ax.set_ylabel(label)
    ax.legend()
    ax.set_ylim(ymin=0)
plt.tight_layout(True)
"""
# plt.savefig("Emulator_validation_1")
