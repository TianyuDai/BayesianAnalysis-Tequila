import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from scipy import interpolate

pp = np.loadtxt("pp2760_pion.txt")
data_RAA = np.loadtxt("data_ATLAS_PbPb2760_0-5central.txt")

cross_section_mb = 28.49
pp_mb = 42

pp_x = pp.T[0]
pp_val = pp.T[1] / 2
pp_err = pp.T[2] / 2

data_RAA_x = data_RAA.T[1]
data_RAA_val = data_RAA.T[3]
data_RAA_err = np.sqrt(data_RAA.T[4]**2+data_RAA.T[5]**2)


n_coll = 960.2
n_coll_err = 96.1

output = np.array([data_RAA_x, data_RAA_val, data_RAA_err]).T
np.savetxt('RAA_data', output)
# data = np.array([data_RAA_x[5:], data_RAA_val[5:], data_RAA_err[5:]]).T


n_dp = [i for i in range(40)]

plt.figure()

for dp in n_dp: 
    # AA = np.loadtxt("PbPb2760_dp%d_pion.txt" %dp)
    AA = np.loadtxt("PbPb2760_pion_val.txt")

    AA_x = AA.T[0]
    AA_val = AA.T[1] / 2
    AA_err = AA.T[2] / 2

    RAA_val = AA_val / pp_val
    RAA_err = RAA_val * np.sqrt((AA_err/AA_val)**2+(pp_err/pp_val)**2)


    output = np.array([AA_x, RAA_val, RAA_err]).T
    
    # np.savetxt('RAA_dp%d' %dp, output)
    np.savetxt('RAA_val', output)
