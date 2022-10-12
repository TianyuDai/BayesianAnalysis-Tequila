import matplotlib.pyplot as plt
import numpy as np
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

for dp in range(31): 
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

    RAA = np.concatenate((np.array([pp_x]).T, np.array([RAA_val]).T, np.array([RAA_err]).T), axis=1)
    np.savetxt('RAA_dp%d'%dp, RAA)

