import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from scipy import interpolate

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

data = np.array([data_RAA_x[5:], data_RAA_val[5:], data_RAA_err[5:]]).T


# n_dp = [6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22]
# n_dp = range(1, 25, 1)
n_dp = [i for i in range(42)]
n_dp.append(60)
n_dp.append(72)
n_dp.append(73)
n_dp.append(88)
# print(n_dp)

plt.figure()

for i in range(1): 
    # AA = np.loadtxt("AA200_dp%d_pion_cs.txt"%i)
    AA = np.loadtxt("AA200_pion_cs_base.txt")
    AA_x = AA.T[0]
    AA_val = AA.T[1] / 2
    AA_err = AA.T[2] / 2

    RAA_val = AA_val / pp_val
    RAA_err = RAA_val * np.sqrt((AA_err/AA_val)**2+(pp_err/pp_val)**2)

    tck = interpolate.splrep(AA_x, RAA_val, s=0)
    cal_RAA_val = interpolate.splev(data_pp_x[16:], tck, der=0)
    tck = interpolate.splrep(AA_x, RAA_err, s=0)
    cal_RAA_err = interpolate.splev(data_pp_x[16:], tck, der=0)

    output = np.array([data_RAA_x[5:], cal_RAA_val, cal_RAA_err]).T
    
    np.savetxt('RAA_base', output)
    # np.savetxt('true_RAA_noInit',cal_RAA_val)
    # np.savetxt('true_RAA_err_noInit', cal_RAA_err)
    plt.errorbar(AA_x, RAA_val, yerr=RAA_err, color='cornflowerblue', alpha=0.5)
    # plt.errorbar(data_pp_x[16:], cal_RAA_val, yerr=cal_RAA_err, color='cornflowerblue')
    plt.errorbar(data_RAA_x, data_RAA_val, yerr=data_RAA_err, color='red', label='data')
    plt.errorbar(data_pp_x[16:], cal_RAA_val, yerr=cal_RAA_err, color='black', label='validation')

plt.legend()
plt.xlim(8, 20)
plt.ylabel('$R_{AA}$')
plt.xlabel('$p_T (GeV)$')
plt.savefig('RAA_validation.pdf')
