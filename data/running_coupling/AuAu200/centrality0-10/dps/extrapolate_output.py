import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from scipy import interpolate

pp = np.loadtxt("pp200_pion.txt")
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

n_coll = 960.2
n_coll_err = 96.1
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

output = np.array([data_RAA_x[6:], data_RAA_val[6:], data_RAA_err[6:]]).T
np.savetxt('RAA_data', output)
# data = np.array([data_RAA_x[5:], data_RAA_val[5:], data_RAA_err[5:]]).T


# n_dp = [6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22]
# n_dp = range(1, 25, 1)
n_dp = [i for i in range(40)]
# n_dp.append(21)
# n_dp.append(72)
# n_dp.append(73)
# n_dp.append(85)
# n_dp.append(88)
# n_dp.append(92)
# n_dp.append(128)
# n_dp.append(134)
# print(n_dp)

plt.figure()

for i in n_dp: 
    AA = np.loadtxt("AA200_dp%d_pion.txt"%i)
    # AA = np.loadtxt("AA200_pion_val.txt")
    # AA_2 = np.loadtxt("AA200_pion_true.txt")
    AA_x = AA.T[0]
    AA_val = AA.T[1] / 2
    AA_err = AA.T[2] / 2

    RAA_val = AA_val / pp_val
    RAA_err = RAA_val * np.sqrt((AA_err/AA_val)**2+(pp_err/pp_val)**2)

    # tck = interpolate.splrep(AA_x, RAA_val, s=0)
    # cal_RAA_val = interpolate.splev(data_pp_x[16:], tck, der=0)
    # tck = interpolate.splrep(AA_x, RAA_err, s=0)
    # cal_RAA_err = interpolate.splev(data_pp_x[16:], tck, der=0)

    # AA2_val = AA_2.T[1] / 2
    # AA2_err = AA_2.T[2] / 2

    # RAA_val_2 = AA2_val / pp_val
    # RAA_err_2 = RAA_val_2 * np.sqrt((AA2_err/AA2_val)**2+(pp_err/pp_val)**2)


    output = np.array([AA_x[6:], RAA_val[6:], RAA_err[6:]]).T
    
    # np.savetxt('RAA_true', output)
    np.savetxt('RAA_dp%d' %i, output)
    # plt.errorbar(AA_x, RAA_val, yerr=RAA_err, color='cornflowerblue', alpha=0.5, label='pre')
    # plt.errorbar(AA_x, RAA_val_2, yerr=RAA_err_2, color='tomato', alpha=0.5, label='new')
    # plt.errorbar(data_pp_x[16:], cal_RAA_val, yerr=cal_RAA_err, color='cornflowerblue')
    # plt.errorbar(data_RAA_x[5:], data_RAA_val[5:], yerr=data_RAA_err[5:], color='red', label='data')
    # plt.errorbar(data_RAA_x[5:], cal_RAA_val[5:], yerr=cal_RAA_err[5:], color='black', label='validation')
"""
plt.legend()
plt.xlim(8, 20)
plt.ylabel('$R_{AA}$')
plt.xlabel('$p_T (GeV)$')
plt.savefig('RAA_validation.pdf')
"""
