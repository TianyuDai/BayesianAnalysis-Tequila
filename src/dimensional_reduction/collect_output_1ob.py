import numpy as np

n_dp = [_ for _ in range(20)]
# n_dp.append(21)
# n_dp.append(72)
# n_dp.append(73)
# n_dp.append(85)
# n_dp.append(88)
# n_dp.append(92)
# n_dp.append(128)
# n_dp.append(134)
# n_dp.append(60)
# n_dp = [6, 7, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21]
# n_dp = [6, 17, 18, 19]
# n_dp = [2, 5, 6, 10, 14, 17, 19, 22, 23, 24, 33, 37, 38, 60, 72, 73, 88]
output = []
output_err = []

for i_dp in n_dp: 
    data_1 = np.loadtxt("../../data/running_coupling/AuAu200/centrality0-10/dps/RAA_dp%d" %i_dp)
    # data_2 = np.loadtxt("../../data/dp_output/centrality40-50/RAA_dp%d" %i_dp)
    i_output_1 = data_1.T[1]
    i_output_err = data_1.T[2]
    # i_output_2 = data_2.T[1]
    # i_output = np.concatenate((i_output_1, i_output_2), axis=0)
    output.append(i_output_1)
    output_err.append(i_output_err)

np.savetxt('../../data/running_coupling/output_1obs', output)
np.savetxt('../../data/running_coupling/output_err_1obs', output_err)

data = np.loadtxt("../../data/running_coupling/AuAu200/centrality0-10/dps/RAA_data")
output = data.T[1]
# output = np.concatenate((output_1, output_2), axis=0)
np.savetxt('../../data/running_coupling/data_exp_1obs', output.T)

output_err = data.T[2]
# output_err = np.concatenate((output_err_1, output_err_2), axis=0)
np.savetxt('../../data/running_coupling/data_exp_err_1obs', output_err.T)

data = np.loadtxt("../../data/running_coupling/AuAu200/centrality0-10/dps/RAA_dp0")
output = data.T[1]
# output = np.concatenate((output_1, output_2), axis=0)
np.savetxt('../../data/running_coupling/data_val_1obs', output.T)

output_err = data.T[2]
# output_err = np.concatenate((output_err_1, output_err_2), axis=0)
np.savetxt('../../data/running_coupling/data_val_err_1obs', output_err.T)
