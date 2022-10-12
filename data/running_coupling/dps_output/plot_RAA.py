import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from scipy import interpolate

# bins = np.linspace(8, 20, 7)
# x = (bins[1:] + bins[:-1]) / 2

pp = np.loadtxt("pp200_hadrons_cross_section_pT.txt")
data_AA = np.loadtxt("data_AuAu200_pions_centrality010.txt")
data_pp = np.loadtxt("pp200_PHENIX_data_hadrons_cross_section.txt")

cross_section_mb = 28.49
pp_mb = 42

pp_x = pp.T[0]
pp_val = pp.T[1] / 2
pp_err = pp.T[2] / 2

data_pp_x = data_pp.T[0]
data_pp_val = data_pp.T[1]
data_pp_err = np.sqrt(data_pp.T[2]**2+data_pp.T[3]**2)

plt.figure()

small_beta = [2, 4, 5, 6, 10, 14, 17, 19, 22, 23, 24]
small_Q0 = [1, 2, 4, 7, 10, 11, 12, 13, 16, 17, 18, 23, 24, 25, 26, 29]
small_ghard = [1, 3, 4, 6, 8, 10, 11, 13, 16, 17, 19, 21, 22, 25, 26, 27]
large_Tstar = [1, 4, 6, 9, 10, 12, 13, 14, 16, 18, 22, 24]
# dp_list = range(30, 43, 1)
dp_list = [6, 60, 72, 73, 88]

for dp in dp_list: 
    AA = np.loadtxt("AA200_dp%d_pion_cs.txt" %dp)

    AA_x = AA.T[0]
    AA_val = AA.T[1] / 2
    AA_err = AA.T[2] / 2

    x = np.linspace(0.25, 50., 100)

    RAA_val = AA_val / pp_val
    RAA_err = RAA_val * np.sqrt((AA_err/AA_val)**2+(pp_err/pp_val)**2)

    # number of collisions for 40-50% centrality
    # n_coll = 124.6

    n_coll = 960.2
    n_coll_err = 96.1
    data_AA_x = data_AA.T[0]
    data_AA_val = data_AA.T[1] * pp_mb / n_coll
    data_AA_err = data_AA_val*np.sqrt((data_AA.T[2] * cross_section_mb)**2/(data_AA.T[1] * cross_section_mb)**2+(n_coll_err/n_coll)**2)

    data_AA_val_log = np.log(data_AA_val)
    tck = interpolate.splrep(data_AA_x, data_AA_val_log, s=0)
    data_AA_interp_log = interpolate.splev(data_pp_x[11:], tck, der=0)
    data_AA_interp = np.exp(data_AA_interp_log)

    data_AA_err_log = np.log(data_AA_err)
    tck = interpolate.splrep(data_AA_x, data_AA_err_log, s=0)
    data_AA_err_interp_log = interpolate.splev(data_pp_x[11:], tck, der=0)
    data_AA_err_interp = np.exp(data_AA_err_interp_log)

    data_RAA_x = data_pp_x[11:]
    data_RAA_val = data_AA_interp / data_pp_val[11:]
    data_RAA_err = data_RAA_val * np.sqrt((data_pp_err[11:]/data_pp_val[11:])**2+(data_AA_err_interp/data_AA_interp)**2)

    # plt.errorbar(pp_x, RAA_val, yerr=RAA_err, alpha=0.5, color='cornflowerblue')
    plt.errorbar(pp_x, RAA_val, yerr=RAA_err, alpha=0.5, label='dp%d'%dp)

plt.errorbar(data_RAA_x, data_RAA_val, yerr=data_RAA_err, label='PHENIX 2013', color='red')

# plt.plot(x, [1 for i in x], color='black')
plt.legend()
plt.xlabel('$p_T$ (GeV/c)')
plt.ylabel('$R_{AA}$')
plt.legend()
plt.ylim(0, 1.)
plt.xlim(8, 20)
plt.title('Tequile, Au+Au 200GeV, 0-10% centrality, $(\Pi^+ + \Pi^-)/2$')
# plt.title('$T^* > 0.35$')
plt.savefig('Tequila_0-10central_RAA_pion_dps_largeRAA.pdf')
