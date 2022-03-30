import subprocess
import numpy as np
import matplotlib.pyplot as plt

from smt.sampling_methods import LHS

# xlimits = np.array([[0.2, 5.], [2., 12.]])
xlimits = np.array([[0.05, 1], [0.05, 0.5], [0.05, 0.5]])
sampling = LHS(xlimits=xlimits)

num = 20
x = sampling(num)

np.savetxt("lhd_sampling_3d_new.txt", x)

# plt.plot(x[:, 0], x[:, 1], "o")
# plt.xlabel("$c$")
# plt.ylabel("$\mu_{cut}$")

# plt.savefig("result/lhd.pdf")
