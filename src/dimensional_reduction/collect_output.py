import numpy as np

n_dp = range(60)
# n_dp = [6, 7, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21]
# n_dp = [6, 17, 18, 19]
output = []
output_err = []

for i_dp in n_dp: 
    data_1 = np.loadtxt("../../data/running_coupling/new_param/AuAu200/centrality0-10/RAA_dp%d" %i_dp)
    data_2 = np.loadtxt("../../data/running_coupling/new_param/AuAu200/centrality20-30/RAA_dp%d" %i_dp)
    data_3 = np.loadtxt("../../data/running_coupling/new_param/PbPb2760/centrality0-5/RAA_dp%d" %i_dp)
    data_4 = np.loadtxt("../../data/running_coupling/new_param/PbPb2760/centrality30-40/RAA_dp%d" %i_dp)
    i_output_1 = data_1.T[1]
    i_output_2 = data_2.T[1]
    i_output_3 = data_3.T[1]
    i_output_4 = data_4.T[1][:-1]
    i_output = np.concatenate((i_output_1, i_output_2, i_output_3, i_output_4), axis=0)
    output.append(i_output)

    i_output_err_1 = data_1.T[2]
    i_output_err_2 = data_2.T[2]
    i_output_err_3 = data_3.T[2]
    i_output_err_4 = data_4.T[2][:-1]
    i_output_err = np.concatenate((i_output_err_1, i_output_err_2, i_output_err_3, i_output_err_4), axis=0)
    output_err.append(i_output_err)

print("combine shape", np.array(output).shape, np.array(output_err).shape)

np.savetxt('../../data/running_coupling/output_4obs', output)
np.savetxt('../../data/running_coupling/output_err_4obs', output_err)

data_1 = np.loadtxt("../../data/running_coupling/AuAu200/centrality0-10/dps/RAA_data")
data_2 = np.loadtxt("../../data/running_coupling/AuAu200/centrality20-30/dps/RAA_data")
data_3 = np.loadtxt("../../data/running_coupling/PbPb2760/centrality0-5/dps/RAA_data")
data_4 = np.loadtxt("../../data/running_coupling/PbPb2760/centrality30-40/dps/RAA_data")
output_1 = data_1.T[1]
output_2 = data_2.T[1]
output_3 = data_3.T[1]
output_4 = data_4.T[1]
# print(output_1)
output = np.concatenate((output_1, output_2, output_3, output_4), axis=0)
print(output_2)
np.savetxt('../../data/running_coupling/data_exp_4obs', output.T)

output_err_1 = data_1.T[2]
output_err_2 = data_2.T[2]
output_err_3 = data_3.T[2]
output_err_4 = data_4.T[2]
output_err = np.concatenate((output_err_1, output_err_2, output_err_3, output_err_4), axis=0)
np.savetxt('../../data/running_coupling/data_exp_err_4obs', output_err.T)

data_1 = np.loadtxt("../../data/running_coupling/new_param/AuAu200/centrality0-10/RAA_true")
data_2 = np.loadtxt("../../data/running_coupling/new_param/AuAu200/centrality20-30/RAA_true")
data_3 = np.loadtxt("../../data/running_coupling/new_param/PbPb2760/centrality0-5/RAA_true")
data_4 = np.loadtxt("../../data/running_coupling/new_param/PbPb2760/centrality30-40/RAA_true")
output_1 = data_1.T[1]
output_2 = data_2.T[1]
output_3 = data_3.T[1]
output_4 = data_4.T[1][:-1]
# print(output_1)
output = np.concatenate((output_1, output_2, output_3, output_4), axis=0)
np.savetxt('../../data/running_coupling/data_val_4obs', output.T)

output_err_1 = data_1.T[2]
output_err_2 = data_2.T[2]
output_err_3 = data_3.T[2]
output_err_4 = data_4.T[2][:-1]
output_err = np.concatenate((output_err_1, output_err_2, output_err_3, output_err_4), axis=0)
np.savetxt('../../data/running_coupling/data_val_err_4obs', output_err.T)
