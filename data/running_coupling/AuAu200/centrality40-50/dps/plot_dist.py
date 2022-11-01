import matplotlib.pyplot as plt
import numpy as np

pp = np.loadtxt("pp200_pion.txt")
pp_ = np.loadtxt("pp200_hadrons_cross_section_pT_newBin.txt")
AA_true = np.loadtxt("AA200_pion_true.txt")
data_AA = np.loadtxt("data_AuAu200_pions_centrality4050.txt")
data_pp = np.loadtxt("pp200_PHENIX_data_hadrons_cross_section.txt")

n_coll = 124.6
n_coll_err = 14.9
pp_mb = 42
cross_section_mb = 28.49

plt.errorbar(pp.T[0], pp.T[1]/2, yerr=pp.T[2]/2, label='pp cal')
# plt.errorbar(pp_.T[0], pp_.T[1]/2, yerr=pp_.T[2]/2, label='pp cal old')
plt.errorbar(data_pp.T[0], data_pp.T[1], yerr=data_pp.T[2], alpha=0.5, label='pp data')
plt.errorbar(AA_true.T[0], AA_true.T[1]/2, yerr=AA_true.T[2], label='AA cal')
plt.errorbar(data_AA.T[0], data_AA.T[1]*pp_mb/n_coll, yerr=data_AA.T[2]*pp_mb/n_coll, alpha=0.5, label='AA data')

plt.legend()
plt.yscale("log")
plt.ylabel('$dN/dp_T$')
plt.xlabel('$p_T$')
plt.xlim(8.25, 19.)
plt.ylim(1e-10, 1e-6)
plt.savefig("pp_AA_dist.pdf")

