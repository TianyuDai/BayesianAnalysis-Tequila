import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from scipy import interpolate

# bins = np.linspace(8, 20, 7)
# x = (bins[1:] + bins[:-1]) / 2

# pp = np.loadtxt("pp200_hadrons_cross_section_pT.txt")

dp_list = range(60)

fig, axes = plt.subplots(2, 2, figsize=(8,8))
fig.subplots_adjust(hspace=0, wspace=0)

data_list = ['data_AuAu200_pions_centrality010.txt', 'data_ATLAS_PbPb2760_0-5central.txt', 'data_AuAu200_pions_centrality20-30.txt', 'data_ATLAS_PbPb2760_30-40central.txt']
label_list = ['PHENIX 2013', 'ATLAS 2015', 'PHENIX 2013', 'ATLAS 2015']

for i, ax, data_name, label in zip(range(4), axes.flat, data_list, label_list): 
    for dp in dp_list: 
        if i == 0: 
            AA = np.loadtxt("AuAu200/centrality0-10/dps/AA200_dp%d_pion.txt" %dp)
        elif i == 2: 
            AA = np.loadtxt("AuAu200/centrality20-30/dps/AA200_dp%d_pion.txt" %dp)
        elif i == 1: 
            AA = np.loadtxt("PbPb2760/centrality0-5/dps/PbPb2760_dp%d_pion.txt" %dp)
        else: 
            AA = np.loadtxt("PbPb2760/centrality30-40/dps/PbPb2760_dp%d_pion.txt" %dp)

        if i == 0 or i == 2: 
            pp = np.loadtxt("pp200_pion.txt")
        else: 
            pp = np.loadtxt("pp2760_pion.txt")
        
        pp_x = pp.T[0]
        pp_val = pp.T[1] / 2
        pp_err = pp.T[2] / 2

        if i == 0 or i == 2: 
            data_AA = np.loadtxt(data_name)
            data_pp = np.loadtxt("pp200_PHENIX_data_hadrons_cross_section.txt")

            cross_section_mb = 28.49
            pp_mb = 42

            data_pp_x = data_pp.T[0]
            data_pp_val = data_pp.T[1]
            data_pp_err = np.sqrt(data_pp.T[2]**2+data_pp.T[3]**2)

            # number of collisions for 40-50% centrality
            # n_coll = 124.6

            if i == 0: 
                n_coll = 960.2
                n_coll_err = 96.1
            else: 
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
            data_RAA_xerr = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 1., 1., 1., 1., 1.]
        else: 
            if i == 1: 
                data_RAA = np.loadtxt("data_ATLAS_PbPb2760_0-5central.txt")
            else: 
                data_RAA = np.loadtxt("data_ATLAS_PbPb2760_30-40central.txt")
            data_RAA_x = data_RAA.T[1]
            data_RAA_val = data_RAA.T[3]
            data_RAA_err = np.sqrt(data_RAA.T[4]**2+data_RAA.T[5]**2)
            data_RAA_xerr = (data_RAA.T[2] - data_RAA.T[0]) / 2


        AA_x = AA.T[0]
        AA_val = AA.T[1] / 2
        AA_err = AA.T[2] / 2

        RAA_val = AA_val / pp_val
        RAA_err = RAA_val * np.sqrt((AA_err/AA_val)**2+(pp_err/pp_val)**2)

        ax.fill_between(AA_x, RAA_val-RAA_err, RAA_val+RAA_err, alpha=0.2, color='cornflowerblue')

    ax.errorbar(data_RAA_x, data_RAA_val, xerr=data_RAA_xerr, yerr=data_RAA_err, fmt='.', label=label, color='red')
    ax.set_ylim(0, 1.5)
    ax.tick_params(direction="in", which='both')
    if i == 1 or i == 3: 
        ax.tick_params(axis='y', which='both', labelleft=False)
    if i == 2 or i == 3: 
        ax.set_xlabel('$p_T$ (GeV/c)')
    if i == 0 or i == 2: 
        ax.set_ylabel('$R_{AA}^{h^\pm}$')

    if i == 0 or i == 2: 
        ax.set_xlim(8., 19)
    else: 
        ax.set_xlim(10, 134)

    ax.legend()
# plt.ylim(0, 1.)
# plt.xlim(8., 20)
# plt.title('Tequile, Au+Au 200GeV, 0-10% centrality, $(\Pi^+ + \Pi^-)/2$')
# plt.title('$T^* > 0.35$')
plt.savefig('Tequila_RAA_pion_dps.pdf')
