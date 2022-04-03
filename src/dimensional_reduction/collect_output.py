import numpy as np

n_dp = range(1, 25, 1)
# n_dp = [6, 7, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21]
# n_dp = [6, 17, 18, 19]
output = []

for i_dp in n_dp: 
    data_1 = np.loadtxt("../../data/dp_output/centrality0-10/RAA_dp%d" %i_dp)
    data_2 = np.loadtxt("../../data/dp_output/centrality40-50/RAA_dp%d" %i_dp)
    i_output_1 = data_1.T[1]
    i_output_2 = data_2.T[1]
    i_output = np.concatenate((i_output_1, i_output_2), axis=0)
    output.append(i_output)

np.savetxt('../../data/dp_output/output', output)
