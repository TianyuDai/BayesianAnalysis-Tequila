import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, '../general/')
from plot_general import hist_1d_2d

simulation = np.loadtxt('../../data/dp_output/output')

X = simulation[:,2]
Y = simulation[:,3]
hist_1d_2d(X, Y, r"$dN_{\rm ch}/d\eta[0-5\%]$", r"$v_2[0-5\%]$")
plt.savefig("../../plots/Check_design_obs_23")
