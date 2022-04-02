import numpy as np

n_dp = range(1, 25, 1)
# n_dp = [6, 7, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21]
# n_dp = [6, 17, 18, 19]
output = []

for i_dp in n_dp: 
    data = np.loadtxt("../../data/dp_output/RAA_dp%d" %i_dp)
    i_output = data.T[1]
    output.append(i_output)

np.savetxt('../../data/dp_output/output', output)
