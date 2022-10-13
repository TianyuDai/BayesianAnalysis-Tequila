import numpy as np

n_dp = [_ for _ in range(43)]
n_dp.append(60)
n_dp.append(72)
n_dp.append(73)
n_dp.append(88)
# n_dp.append(60)
# n_dp = [6, 7, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21]
# n_dp = [6, 17, 18, 19]
n_dp = [2, 5, 6, 10, 14, 17, 19, 22, 23, 24, 33, 37, 38, 60, 72, 73, 88]
output = []

for i_dp in n_dp: 
    data_1 = np.loadtxt("../../data/running_coupling/dps_output/RAA_dp%d" %i_dp)
    # data_2 = np.loadtxt("../../data/dp_output/centrality40-50/RAA_dp%d" %i_dp)
    i_output_1 = data_1.T[2]
    # i_output_2 = data_2.T[1]
    # i_output = np.concatenate((i_output_1, i_output_2), axis=0)
    output.append(i_output_1)

np.savetxt('../../data/running_coupling/output_small_beta_err', output)
