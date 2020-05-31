import subprocess
# git clone https://github.com/keweiyao/BayesExample
# Python3 
# python package: sklearn, numpy, matplotlib scipy, emcee (for MCMC)
# R
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process import kernels
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from scipy import stats
import emcee
import numpy as np


from scipy.integrate import quad
from scipy.special import gammaln

# In this example, we use a very simple model of jet-quenching to practice all the
# all the techniques we learned above:
# Given a measurement: yexp +/- ystat +/- ysys,
#       and a model M : parameters --> predictions
# 1) Make a parameter design on which we run the computer model
# 2) Use Principal Component Analysis (PCA) to do dimension reduction
# 3) Build Gaussian Process emulators (GP) and train on the design
# 4) Construct proir, likelihood and posterior function
# 5) Use MCMC to marginalize the posterior distribution
# 6) Analyse the posterior distribution of the parameters


########### A simple model that calculates a "R_AA" ########################
# Baseline of particle production: dN0/dpT ~ pT/(3^2 + pT^2)^3

# "Experimental data"
# Assume the model is perfect, and the truth values of A and B are 1.0 and 0.5
### A = 1 , B = 0.5
truth = [1.]
# The Measurement measure the truth Raa, subject to limited statistics and systematic bias
# yexp = y_true + ystat + ysys
# Default: 10% relative statistical uncertainty, ystat=0

################Step 1: Get the Measurement########################
#  Choose the magnitudes of Stat and Sys error, (recommanded 10%)
#  1) plot the true Raa v.s. pT
#  2) plot the experimental data with stat and sys error
#
###################################################################

data = np.loadtxt("AuAu200_RAA_cut15_coef1")
pT = data.T[0]
pTbin = np.linspace(10., 30., 11)
d_pT = (pTbin[1:]-pTbin[:-1])/2

ytruth = data.T[1]
yexp = ytruth
ystat = data.T[2]
print(pT.shape, d_pT.shape, yexp.shape, ystat.shape)

plt.figure()
# plot True Raa here
# plt.plot(pT, ytruth, 'k.', label='Truth') 

# plot Raa measurement with stat errorbars
plt.errorbar(x=pT, xerr=d_pT, y=yexp, yerr=ystat, fmt='rD', label=r"$Truth \pm$ stat") 
plt.text(10, 0.05, 'Au+Au 200GeV, 0~10%')
# plot sys errorband, y1:lowerbounds, y2: higherbounds
# plt.fill_between(x=pT, y1=yexp-ysys, y2=yexp+ysys, 
#                 color='r', alpha=.3, label=r"$\pm$ sys")

plt.ylim(0,1)
plt.semilogx()
plt.xlabel(r'$p_T$ [GeV]', fontsize=15)
plt.ylabel(r'$R_{AA}$', fontsize=15)
plt.legend()

plt.savefig('data_RAA.pdf')

################Step 2: ###################################
# Make a design over the parameter space (A, B)
# 1) What is a reasonable range of the prior?
# 2) Generate the design and rescale it to the desired range
#    Hint: linear rescale x from (0,1) to y from (a,b):
#           y = (1-x)*a + x*b
# 3) Run model() on each design points
#    The design matrix model_data should have a shape: 
#           N_design x N_pT
#[amin, amax]

design = np.array([[0.5], [1.], [2.], [10.]])
print(design)
data_1 = np.loadtxt("AuAu200_RAA_cut15_coef05")
data_2 = np.loadtxt("AuAu200_RAA_cut15_coef1")
data_3 = np.loadtxt("AuAu200_RAA_cut15_coef2")
data_4 = np.loadtxt("AuAu200_RAA_cut15_coef10")
model_data = []
model_data.append(data_1.T[1])
model_data.append(data_2.T[1])
model_data.append(data_3.T[1])
model_data.append(data_4.T[1])
model_data = np.array(model_data)


###### Step 3: apply PCA ######################################
# We don't need a separate GP for each pT point
# 1) Try keeping different number of principal components (npc).
#    How many pc(s) do you think is enought for this exercises?
# 2) Take a look at the feature of each pc (figure 3).
#    What each pc does in terms of decomposing the data?
# 3) Look at the coorelation between PC1 and PC2 (figure 2),
#    are they completely uncorrelated? Combine with your
#    observations from (figure 3), can you explain what
#    causes the correlation?
################################################################
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

# ...
plt.figure(figsize=(15,4))
F_r = np.cumsum(pca.explained_variance_ratio_)
plt.subplot(1,3,1)
plt.plot(range(len(F_r)),F_r,'-o')
ticks = np.arange(0, len(F_r), 1)
labels = ticks + 1
plt.xticks(ticks, labels)
plt.title('Fraction of Variance Explained')
plt.xlabel('Number of Components')
plt.ylabel('F_r')

plt.subplot(1,3,2)
plt.scatter(Z[:, 0], Z[:, 1])
plt.xlabel('PC1', fontsize=15)
plt.ylabel('PC2', fontsize=15)

plt.subplot(1,3,3)
for i, (comp, color) in \
        enumerate(zip(pca.components_, 'rgb')):
    plt.plot(pT, comp, color=color, label='PC {}'.format(i+1))
plt.xlabel('$p_T$', fontsize=15)
plt.ylabel('$Features$', fontsize=15)
plt.legend()
plt.tight_layout(True)
plt.savefig("pca.pdf")

######## Step 4-1: Building Emulators #############################
# Using an Exp-Squared kernel + a white kernel (accounting 
# for numerical error of model calculations)
# 1) Put in initial length scales for param A and B
# 2) Put in reasonable lenght scales bounds for optimization
# 3) Fit separate emulatior to each principal component.
#    Take a look at the optimized hyper-parameters.
#    What do they mean?
"""
kernel = (
    1. * kernels.RBF(
        length_scale=[1, 1],
        length_scale_bounds=[(.1,10), (.1, 10)]
    )  
    + kernels.WhiteKernel(.1)
)
"""
kernel = (
    1. * kernels.RBF(
        length_scale=1,
        length_scale_bounds=(.1,10)
    )  
    + kernels.WhiteKernel(.1)
)
# Build and train each GP
gps = [ GPR(kernel=kernel, n_restarts_optimizer=10) 
        for i in range(npc) ]
for i, gp in enumerate(gps):
    gp.fit(design, Z[:,i])
    print('RBF: ', gp.kernel_.get_params()['k1'])
    print('White: ', gp.kernel_.get_params()['k2'])

### Step 4-2: Validating the emulators #######################
# It is important to validate the performance of emulators to
# make sure they behave as expected.
# 1) Pick 6 random combinations of A and B. Compare the
#    emulators prediction and the model calculations.
#    Do they agree? 
fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True)
for a in [0.6, 0.8, 0.75, 1.2, 1.5, 1.8, 1.9, 2.8, 3.2, 4.6, 5.7, 6.2, 7.8, 9.1]:
    # GP prediction
    z = np.array([gp.predict([[a]])[0] for gp in gps])
    pred = np.dot(z, Trans_Matrix[:z.shape[-1]])
    pred += scaler.mean_
    # model calcuatlion
    # calc = Model(pT, a, b)
    
    # ax1.plot(pT, calc, 'ro', alpha=0.7)
    ax1.plot(pT, pred, 'b--', alpha=0.7)
    # ax2.plot(pT, (pred-calc)/calc, 'b--', alpha=0.7)

ax1.semilogx()
ax1.set_xlabel(r'$p_T$ [GeV]', fontsize=15)
ax1.set_ylabel(r'$R_{AA}$', fontsize=15)
ax2.set_ylim(-.2, .2)
ax2.set_xlabel(r'$p_T$ [GeV]', fontsize=15)
ax2.set_ylabel('relative error', fontsize=15)

plt.tight_layout(True)
plt.savefig("validation.pdf")

##### Helper functions for this block ###################
from scipy.linalg import lapack
# calculate the log of Gaussian density with 
# residual dy = y-mu and covariance matrix cov.
# - 1/2 * dy^T * cov^[-1] * dy - 1/2*ln(|cov|)
def lnLL(dy, cov):
    L, info = lapack.dpotrf(cov, clean=False)
    alpha, info = lapack.dpotrs(L, dy)
    return -.5*np.dot(dy, alpha)-np.log(L.diagonal()).sum()

# Transform a covariance matrix from the PC space 
# back to the physical space
def transform_cov(std):
    cov = np.matmul(Trans_Matrix[:npc].T*std**2, 
                    Trans_Matrix[:npc])\
        + Residual_Cov 
    return cov

####### Step 5: Construct the posterior #################
# Remember that from Bayes' Theorem:
#      Posterior  = prior * likelihood
# and:
#      ln(Posterior) = ln(prior) + ln(likelihood)
# and:
#      theta = [A, B]
# 1) Complete the returns of the prior "prior_ln_pdf(theta)"
# 2) The sys-error is correlated, while the stat one is not
#    We provide two types of covariance matrixx
#        2.1) cov_exp1 treats sys-error as uncorrelated
#        2.2) cov_exp2 treats sys-error as correlated
#    Start with 2.1) and later try 2.2) to see the effects
#    on the posterior distribution of A and B.
# 3) Complete the likelihood_ln_pdf(theta) function
def prior_ln_pdf(theta):
    if (theta<ranges[0]).any() or (theta>ranges[1]).any():
        return -np.inf
    else:
        return 0.

# Pick your experimental covariance matrix
Assume_SysError_Corr = True
cov_exp = np.diag(ystat**2)

def likelihood_ln_pdf(theta):
    z, stdz = np.array([gp.predict([theta], return_std=True) for gp in gps]).T[0]
    pred = np.dot(z, Trans_Matrix[:z.shape[-1]])
    pred += scaler.mean_
    cov_emulator = transform_cov(std=stdz)
    dy = pred-yexp
    cov = cov_exp + cov_emulator
    return lnLL(dy, cov)

# Finally ln(Posterior) = ln(prior) + ln(likelihood)
def posterior_ln_pdf(theta):
    ln_pr = prior_ln_pdf(theta)
    ln_like = likelihood_ln_pdf(theta) 
    return ln_pr + ln_like


######### Step 6: Run MCMC ###########################
# Fill 1) the number of samples 2) burnin steps
# 3) dimsional of the problem 4) number of mcmc walkers
# Hint: this may take a while, so start with smaller numbers
nsteps = 2000
nburnin = 100
ndim = 1
nwalkers = 20
sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior_ln_pdf)
ranges = np.array([0.5, 10.])
p0 = np.random.rand(nwalkers, ndim)
p0 = p0*ranges[0] +  p0*ranges[1]

out_post = sampler.run_mcmc(p0, nsteps)
samples = sampler.chain[:, nburnin:, :].reshape((-1, ndim))

##### Step 7: Analyze the posterior distribution ########
# 1) Run this block and plot the posterior distribution
# 2) Does the posterior fairly estimates the true values (red)?
# 3) How does the posterior change it we take into account the
#    correlation among the sys-error?
plt.figure(figsize=(5,5))
names = [r'$c$']
plt.hist(samples, bins=40, range=ranges, histtype='step', density=True)
plt.xlabel(names)
plt.axvline(x=truth[0], color='r', linewidth=1)
plt.xlim(ranges)
plt.savefig("corr-results.pdf")

# predicting observables
param_samples = samples[ np.random.choice(range(len(samples)),50), :]
z  = np.array([gp.predict(param_samples) for gp in gps]).T
pred = np.dot(z, Trans_Matrix[:z.shape[-1]])
pred += scaler.mean_
plt.figure()
for i, y in enumerate(pred):
    plt.plot(pT, y, 'b-', alpha=0.15, label="Posterior" if i==0 else '')
plt.errorbar(pT, yexp, yerr=ystat, xerr=d_pT, fmt='ro', label="Measurements")
plt.ylim(0,1)
plt.semilogx()
plt.xlabel(r'$p_T$ [GeV]', fontsize=15)
plt.ylabel(r'$R_{AA}$', fontsize=15)
plt.legend()
plt.savefig("prediction.pdf")

