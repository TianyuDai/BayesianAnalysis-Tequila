import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from scipy import interpolate

# bins = np.linspace(8, 20, 7)
# x = (bins[1:] + bins[:-1]) / 2

# pp = np.loadtxt("pp200_hadrons_cross_section_pT.txt")
pp = np.loadtxt("pp2760_pion.txt")
data_RAA = np.loadtxt("data_ATLAS_PbPb2760_0-5central.txt")

cross_section_mb = 28.49
pp_mb = 42

pp_x = pp.T[0]
pp_val = pp.T[1] / 2
pp_err = pp.T[2] / 2

# number of collisions for 40-50% centrality
# n_coll = 124.6

n_coll = 960.2
n_coll_err = 96.1

data_RAA_x = data_RAA.T[1]
data_RAA_val = data_RAA.T[3]
data_RAA_err = np.sqrt(data_RAA.T[4]**2+data_RAA.T[5]**2)

plt.figure()


dp_list = range(40)

for dp in dp_list:
 
    AA = np.loadtxt("PbPb2760_dp%d_pion.txt" %dp)

    AA_x = AA.T[0]
    AA_val = AA.T[1] / 2
    AA_err = AA.T[2] / 2

    RAA_val = AA_val / pp_val
    RAA_err = RAA_val * np.sqrt((AA_err/AA_val)**2+(pp_err/pp_val)**2)

    # plt.errorbar(pp_x, RAA_val, yerr=RAA_err, alpha=0.5, color='cornflowerblue')
    plt.fill_between(pp_x, RAA_val-RAA_err, RAA_val+RAA_err, alpha=0.2, color='cornflowerblue')
    # plt.errorbar(pp_x, RAA_val, yerr=RAA_err, label='dp%d'%dp)



plt.errorbar(data_RAA_x, data_RAA_val, yerr=data_RAA_err, label='PHENIX 2013', color='red')


# AA_upper = np.loadtxt("PbPb2760_pion_Q01.6.txt")
# AA_true = np.loadtxt("PbPb2760_pion_Q03.1.txt")
# AA_lower = np.loadtxt("PbPb2760_pion_Q02.5.txt")

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


AA_true_val = AA_true.T[1] / 2
AA_true_err = AA_true.T[2] / 2

RAA_true_val = AA_true_val / pp_val
RAA_true_err = RAA_true_val * np.sqrt((AA_true_err/AA_true_val)**2+(pp_err/pp_val)**2)

# plt.errorbar(data_AA_x, RAA_upper_val, yerr=RAA_upper_err, label='$\\tau_0 = 0.2$ fm/c')
plt.errorbar(data_RAA_x, RAA_true_val, yerr=RAA_true_err, label='$\\tau_0 = 0.5$ fm/c')
# plt.errorbar(data_AA_x, RAA_lower_val, yerr=RAA_lower_err, label='$\\tau_0 = 0.8$ fm/c')

# plt.fill_between(data_RAA_x, RAA_upper_val-RAA_upper_err, RAA_upper_val+RAA_upper_err, label='$Q_0 = 1.6$ GeV', color='tomato', alpha=0.5)
# plt.fill_between(data_RAA_x, RAA_lower_val+RAA_lower_err, RAA_lower_val-RAA_lower_err, label='$Q_0 = 2.5$ GeV', color='mediumseagreen', alpha=0.5)
# plt.fill_between(data_RAA_x, RAA_true_val-RAA_true_err, RAA_true_val+RAA_true_err, label='$Q_0 = 3.1$ GeV', color='cornflowerblue', alpha=0.5)
"""
# plt.plot(x, [1 for i in x], color='black')
plt.xlabel('$p_T$ (GeV/c)')
plt.ylabel('$R_{AA}$')
plt.legend()
plt.ylim(0, 1.)
plt.xlim(10, 120)
plt.title('Tequile, Pb+Pb 2760GeV, 0-5% centrality, $(\Pi^+ + \Pi^-)/2$')
# plt.title('$T^* > 0.35$')
plt.savefig('Tequila_0-5central_RAA_pion_dps.pdf')
