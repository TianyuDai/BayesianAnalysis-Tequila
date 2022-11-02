import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from scipy import interpolate

# bins = np.linspace(8, 20, 7)
# x = (bins[1:] + bins[:-1]) / 2

# pp = np.loadtxt("pp200_hadrons_cross_section_pT.txt")
pp = np.loadtxt("pp200_pion.txt")
data_AA = np.loadtxt("data_AuAu200_pions_centrality20-30.txt")
data_pp = np.loadtxt("pp200_PHENIX_data_hadrons_cross_section.txt")

cross_section_mb = 28.49
pp_mb = 42

pp_x = pp.T[0]
pp_val = pp.T[1] / 2
pp_err = pp.T[2] / 2

data_pp_x = data_pp.T[0]
data_pp_val = data_pp.T[1]
data_pp_err = np.sqrt(data_pp.T[2]**2+data_pp.T[3]**2)

# number of collisions for 40-50% centrality
# n_coll = 124.6

n_coll = 377.6
n_coll_err = 36.4
data_AA_x = data_AA.T[0]
data_AA_val = data_AA.T[1] * pp_mb / n_coll
data_AA_err = data_AA_val*np.sqrt((data_AA.T[2] * cross_section_mb)**2/(data_AA.T[1] * cross_section_mb)**2+(n_coll_err/n_coll)**2)

data_pp_val_log = np.log(data_pp_val)
tck = interpolate.splrep(data_pp_x, data_pp_val_log, s=0)
data_pp_interp_log = interpolate.splev(data_AA_x, tck, der=0)
data_pp_interp = np.exp(data_pp_interp_log)

data_pp_err_log = np.log(data_pp_err)
tck = interpolate.splrep(data_pp_x, data_pp_err_log, s=0)
data_pp_err_interp_log = interpolate.splev(data_AA_x, tck, der=0)
data_pp_err_interp = np.exp(data_pp_err_interp_log)

data_RAA_x = data_AA_x
data_RAA_val = data_AA_val / data_pp_interp
data_RAA_err = data_RAA_val * np.sqrt((data_pp_err_interp/data_pp_interp)**2+(data_AA_err/data_AA_val)**2)

plt.figure()
"""
dp_list = range(30, 40)

for dp in dp_list: 
    AA = np.loadtxt("AA200_dp%d_pion.txt" %dp)

    AA_x = AA.T[0]
    AA_val = AA.T[1] / 2
    AA_err = AA.T[2] / 2

    RAA_val = AA_val / pp_val
    RAA_err = RAA_val * np.sqrt((AA_err/AA_val)**2+(pp_err/pp_val)**2)

    # plt.errorbar(pp_x, RAA_val, yerr=RAA_err, alpha=0.5, color='cornflowerblue')
    # plt.fill_between(pp_x, RAA_val-RAA_err, RAA_val+RAA_err, alpha=0.2, color='cornflowerblue')
    plt.errorbar(pp_x, RAA_val, yerr=RAA_err, label='dp%d'%dp)
"""
plt.errorbar(data_RAA_x, data_RAA_val, yerr=data_RAA_err, label='PHENIX 2013', color='red')

# AA_upper = np.loadtxt("AA200_pion_tau0_0.2.txt")
AA_true = np.loadtxt("AA200_centrality20-30_pion_val.txt")
# AA_lower = np.loadtxt("AA200_pion_tau0_0.8.txt")

# AA_x = AA.T[0]
"""
AA_upper_val = AA_upper.T[1] / 2
AA_upper_err = AA_upper.T[2] / 2

RAA_upper_val = AA_upper_val / pp_val
RAA_upper_err = RAA_upper_val * np.sqrt((AA_upper_err/AA_upper_val)**2+(pp_err/pp_val)**2)

AA_lower_val = AA_lower.T[1] / 2
AA_lower_err = AA_lower.T[2] / 2

RAA_lower_val = AA_lower_val / pp_val
RAA_lower_err = RAA_lower_val * np.sqrt((AA_lower_err/AA_lower_val)**2+(pp_err/pp_val)**2)
"""
AA_true_val = AA_true.T[1] / 2
AA_true_err = AA_true.T[2] / 2

RAA_true_val = AA_true_val / pp_val
RAA_true_err = RAA_true_val * np.sqrt((AA_true_err/AA_true_val)**2+(pp_err/pp_val)**2)

# plt.errorbar(data_AA_x, RAA_upper_val, yerr=RAA_upper_err, label='$\\tau_0 = 0.2$ fm/c')
plt.errorbar(data_AA_x, RAA_true_val, yerr=RAA_true_err, label='valid')
# plt.errorbar(data_AA_x, RAA_lower_val, yerr=RAA_lower_err, label='$\\tau_0 = 0.8$ fm/c')

# plt.fill_between(data_AA_x, RAA_upper_val-RAA_upper_err, RAA_upper_val+RAA_upper_err, label='$\\tau_0 = 0.2$ fm/c', color='tomato', alpha=0.5)
# plt.fill_between(data_AA_x, RAA_true_val-RAA_true_err, RAA_true_val+RAA_true_err, label='$\\tau_0 = 0.5$ fm/c', color='cornflowerblue', alpha=0.5)
# plt.fill_between(data_AA_x, RAA_lower_val+RAA_lower_err, RAA_lower_val-RAA_lower_err, label='$\\tau_0 = 0.8$ fm/c', color='mediumseagreen', alpha=0.5)

# plt.plot(x, [1 for i in x], color='black')
plt.xlabel('$p_T$ (GeV/c)')
plt.ylabel('$R_{AA}$')
plt.legend()
plt.ylim(0, 1.)
plt.xlim(8.25, 19)
plt.title('Tequile, Au+Au 200GeV, 20-30% centrality, $(\Pi^+ + \Pi^-)/2$')
# plt.title('$T^* > 0.35$')
plt.savefig('Tequila_20-30central_RAA_pion_val.pdf')
