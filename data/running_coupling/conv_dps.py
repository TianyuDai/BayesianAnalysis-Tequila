import numpy as np

beta_1 = []
beta_2 = []
beta_3 = []
Tstar = []
Q0 = []
ghard = []

dps_list = [_ for _ in range(43)]
dps_list.append(60)
dps_list.append(72)
dps_list.append(73)
dps_list.append(88)

dps_list = [2, 5, 6, 10, 14, 17, 19, 22, 23, 24, 33, 37, 38, 60, 72, 73, 88]

with open("design_points_main_AuAu-200.dat") as file: 
    for line in file: 
        line = line.rstrip()
        if "idx" in line: 
            continue
        dp_point = line.split(',')

        if not int(dp_point[0]) in dps_list: 
            continue

        beta_1.append(float(dp_point[1]))
        beta_2.append(float(dp_point[2]))
        beta_3.append(float(dp_point[3]))
        Tstar.append(float(dp_point[4]))
        Q0.append(float(dp_point[5]))
        ghard.append(float(dp_point[6]))

lhd_sampling = np.concatenate((np.array([beta_1]).T, np.array([beta_2]).T, np.array([beta_3]).T, np.array([Tstar]).T, np.array([Q0]).T, np.array([ghard]).T), axis=1)

np.savetxt('lhd_sampling_6d_small_beta.txt', lhd_sampling)
