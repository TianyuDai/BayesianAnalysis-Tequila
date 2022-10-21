import numpy as np

n_dp = range(20)
# n_dp = [6, 7, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21]
# n_dp = [6, 17, 18, 19]
output = []
output_err = []

for i_dp in n_dp: 
    data_1 = np.loadtxt("../../data/running_coupling/centrality0-10/dps/RAA_dp%d" %i_dp)
    data_2 = np.loadtxt("../../data/running_coupling/centrality40-50/dps/RAA_dp%d" %i_dp)
    i_output_1 = data_1.T[1]
    i_output_2 = data_2.T[1]
    i_output = np.concatenate((i_output_1, i_output_2), axis=0)
    output.append(i_output)

    i_output_err_1 = data_1.T[2]
    i_output_err_2 = data_2.T[2]
    i_output_err = np.concatenate((i_output_err_1, i_output_err_2), axis=0)
    output_err.append(i_output_err)

print("combine shape", np.array(output).shape, np.array(output_err).shape)

np.savetxt('../../data/running_coupling/output_2obs', output)
np.savetxt('../../data/running_coupling/output_err_2obs', output_err)

data_1 = np.loadtxt("../../data/running_coupling/centrality0-10/dps/RAA_data")
data_2 = np.loadtxt("../../data/running_coupling/centrality40-50/dps/RAA_data")
output_1 = data_1.T[1]
output_2 = data_2.T[1]
# print(output_1)
output = np.concatenate((output_1, output_2), axis=0)
np.savetxt('../../data/running_coupling/data_exp_2obs', output.T)

output_err_1 = data_1.T[2]
output_err_2 = data_2.T[2]
output_err = np.concatenate((output_err_1, output_err_2), axis=0)
np.savetxt('../../data/running_coupling/data_exp_err_2obs', output_err.T)

data_1 = np.loadtxt("../../data/running_coupling/centrality0-10/dps/RAA_true")
data_2 = np.loadtxt("../../data/running_coupling/centrality40-50/dps/RAA_true")
output_1 = data_1.T[1]
output_2 = data_2.T[1]
# print(output_1)
output = np.concatenate((output_1, output_2), axis=0)
np.savetxt('../../data/running_coupling/data_val_2obs', output.T)

output_err_1 = data_1.T[2]
output_err_2 = data_2.T[2]
output_err = np.concatenate((output_err_1, output_err_2), axis=0)
np.savetxt('../../data/running_coupling/data_val_err_2obs', output_err.T)
