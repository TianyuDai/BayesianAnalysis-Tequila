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
from pyDOE import lhs
from multiprocessing import Pool, cpu_count
import emcee



"""
ParameterLabels = ['$N$',
                   r'$\frac{\eta}{s}(a)$', 
                   r'$\frac{\eta}{s}(b)$ [GeV${}^{-1}$]', 
                   r'$\frac{\eta}{s}(c)$[GeV${}^{-1}$]', 
                   r'$\frac{\eta}{s}(b)$ [GeV]']
ParameterRanges = {'N': [6,15], 
                   'a': [.02, 0.2], 
                   'b': [-1,4], 
                   'c': [0,10], 
                   'd': [0.1, 0.4] }
Xdim = len(ParameterLabels) # dimension of the parameter space
Ndesign = 20

# range of the design, save to file
ranges = np.array([ParameterRanges[it] for it in ParameterRanges.keys()])
# First make a Latin-Hypercube design within the Xdim unit cube [0,1]^Xdim
np.random.seed(1)
unit = lhs(ranges.shape[0], samples=Ndesign, criterion='maximin')
# Then, rescale the design tot he desired range
design = ranges[:,0] + unit*(ranges[:,1]-ranges[:,0])

print(design.shape)


simulation = np.array([[i]*9 for i in range(20)])
print(simulation)
"""

simulation = np.loadtxt('../../data/dp_output/output')
use_NL = True
Y_model = np.sqrt(simulation) if use_NL else simulation
# print(Y_model.shape)
# Y_model = np.loadtxt('../../data/dp_output/output')

SS  =  StandardScaler(copy=True)
Npc = 3
pca = PCA(copy=True, whiten=True, svd_solver='full')
# Keep only the first `npc` principal components
pc_tf_data = pca.fit_transform(SS.fit_transform(Y_model)) [:,:Npc]

# The transformation matrix from PC to Physical space
inverse_tf_matrix = pca.components_ * np.sqrt(pca.explained_variance_[:, np.newaxis]) * SS.scale_ 
inverse_tf_matrix = inverse_tf_matrix[:Npc,:]

np.savetxt('../../data/inverse_tf_matrix', inverse_tf_matrix)
# print('original', Y_model[0])
# print('after transformation', pc_tf_data[0])
mean = np.array([pc_tf_data[0]])
# print(mean@inverse_tf_matrix + np.array(SS.mean_).reshape(1,-1))
np.savetxt('../../data/PCA_transformed_data', pc_tf_data)

"""
design_max = ranges[:,1]
design_min = ranges[:,0]
# The range of the design is an important reference length scale when we train the emulator 
design_ptp = design_max - design_min
"""
design_ptp = np.array([1-0.05, 0.5-0.05, 0.5-0.05])
design = np.loadtxt('../../data/lhd_sampling_3d.txt')
# design = np.array([[i]*3 for i in range(200)])
print(design.shape)
# pc_tf_data = np.loadtxt('../../data/PCA_transformed_data')
# pc_tf_data = np.loadtxt('../../data/dp_output/output')
print(pc_tf_data.shape)

overide = True
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
use_NL = True
def predict_observables(model_parameters, diag_std=False):
    """Predicts the observables for any model parameter value using the trained emulators.
    
    Parameters
    ----------
    Theta_input : Model parameter values.
    Return
    ------
    Mean value and full error covaraiance matrix of the prediction is returened. """
   
    # use_NL = True
    # Xdim = 3
    # inverse_tf_matrix = np.loadtxt('../../data/inverse_tf_matrix')
 
    mean=[]
    variance=[]
    theta=np.array(model_parameters).flatten()
    if len(theta)!=Xdim:
        raise TypeError('The input model_parameters array does not have the right dimensions')
    else: 
        theta=np.array(theta).reshape(1,Xdim)
        for i in range(Npc):
            mn,std=Emulators[i].predict(theta, return_std=True)
            mean.append(mn)
            variance.append(std**2)
    mean=np.array(mean).reshape(1,-1)
    # print('mean', mean)
    # mean=np.array([pc_tf_data[0]])
    # print('mean', mean)
    # print('mean shape', mean.shape)
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

# design_min = np.array([0.05, 0.05, 0.05])
# inverse_tf_matrix = np.loadtxt('../../data/PCA_transform_matrix')

# randomly sample another 2 parametr points from the parameter space
# np.random.seed(9)
# X_validation = design_min + np.random.rand(2, Xdim)*design_ptp
# X_validation = np.array([[0.596, 1.51, 3.99]])
X_validation = np.array([[0.3, 0.3, 0.3]])

# Next, get the emulator prediction and uncertainty
A = np.array([predict_observables(it, diag_std=True) for it in X_validation])
Y_predicted = A[:,0,:]
Y_std = A[:,1,:]
np.savetxt('../../data/data_predicted', Y_predicted)


y_exp = np.loadtxt('../../data/RAA_data')
y_err = np.loadtxt('../../data/RAA_data_err')
y_exp_variance = np.diag(y_err**2)

def log_prior(model_parameters):
    """Evaluvate the prior at model prameter values. 
    If all parameters are inside bounds function will return 0 otherwise -inf"""
    X = np.array(model_parameters).reshape(1,-1)
    lower = np.all(X >= design_min)
    upper = np.all(X <= design_max)
    if (lower and upper):
        lp=0
    # lp = np.log(st.beta.pdf(X,5,1,dsgn_min_ut.reshape(1,-1),(dsgn_max_ut-dsgn_min_ut).reshape(1,-1))).sum()
    else:
        lp = -np.inf
    return lp

def mvn_loglike(y, cov):
    """
    Evaluate the multivariate-normal log-likelihood for difference vector `y`
    and covariance matrix `cov`:

        log_p = -1/2*[(y^T).(C^-1).y + log(det(C))] + const.

    The likelihood is NOT NORMALIZED, since this does not affect MCMC.  The
    normalization const = -n/2*log(2*pi), where n is the dimensionality.

    Arguments `y` and `cov` MUST be np.arrays with dtype == float64 and shapes
    (n) and (n, n), respectively.  These requirements are NOT CHECKED.

    The calculation follows algorithm 2.1 in Rasmussen and Williams (Gaussian
    Processes for Machine Learning).

    """
    # Compute the Cholesky decomposition of the covariance.
    # Use bare LAPACK function to avoid scipy.linalg wrapper overhead.
    L, info = lapack.dpotrf(cov, clean=False)

    if info < 0:
        raise ValueError(
            'lapack dpotrf error: '
            'the {}-th argument had an illegal value'.format(-info)
        )
    elif info < 0:
        raise np.linalg.LinAlgError(
            'lapack dpotrf error: '
            'the leading minor of order {} is not positive definite'
            .format(info)
        )

    # Solve for alpha = cov^-1.y using the Cholesky decomp.
    alpha, info = lapack.dpotrs(L, y)

    if info != 0:
        raise ValueError(
            'lapack dpotrs error: '
            'the {}-th argument had an illegal value'.format(-info)
         )

    if np.all(L.diagonal()>0):
        return -.5*np.dot(y, alpha) - np.log(L.diagonal()).sum()
    else:
        return -.5*np.dot(y, alpha) - np.log(np.abs(L.diagonal())).sum()
        print(L.diagonal())
        raise ValueError(
            'L has negative values on diagonal {}'.format(L.diagonal())
        )

def log_posterior(model_parameters):
    mn, var = predict_observables(model_parameters)
    delta_y = mn - y_exp
    delta_y = delta_y.flatten()   
    total_var = var + y_exp_variance
    return log_prior(model_parameters) + mvn_loglike(delta_y,total_var)


Xdim = 3
nwalkers = 10*Xdim  # number of MCMC walkers
nburn = 500 # "burn-in" period to let chains stabilize
nsteps = 2000  # number of MCMC steps to take
# filename = data_path(name+".h5")

design_min = np.array([0.05, 0.05, 0.05])
design_max = np.array([1., 0.5, 0.5])

#backend = emcee.backends.HDFBackend(filename)
starting_guesses = design_min + (design_max - design_min) * np.random.rand(nwalkers, Xdim)
#print(starting_guesses)
print("MCMC sampling using emcee (affine-invariant ensamble sampler) with {0} walkers".format(nwalkers))
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, Xdim, log_posterior)
    print('burn in sampling started')    
    pos = sampler.run_mcmc(starting_guesses, nburn, progress=True, store=True)
    print("Mean acceptance fraction: {0:.3f} (in total {1} steps)".format(
                        np.mean(sampler.acceptance_fraction), nwalkers*nburn))
    print('Burn in completed.')
    print("Now running the samples")
    sampler.run_mcmc(initial_state=None, nsteps=nsteps, progress=True, tune=False)  
    print("Mean acceptance fraction: {0:.3f} (in total {1} steps)".format(
                        np.mean(sampler.acceptance_fraction), nwalkers*nsteps))
        
    # discard burn-in points and flatten the walkers; the shape of samples is (nwalkers*nsteps, Xdim)
    #samples = backend.get_chain(flat=True, discard=nburn)
    samples = sampler.get_chain(flat=True, discard=nburn)

np.savetxt('../../data/MCMC_samples', samples)
