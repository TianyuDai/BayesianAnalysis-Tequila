import numpy as np

data = np.loadtxt("data/pp200_cross_section.txt")

data1 = np.loadtxt("data/AuAU200_cross_section_cut15_coef05.txt")
data2 = np.loadtxt("data/AuAU200_cross_section_cut15_coef1.txt")
data3 = np.loadtxt("data/AuAU200_cross_section_cut15_coef2.txt")
data4 = np.loadtxt("data/AuAU200_cross_section_cut15_coef10.txt")

err = [0.02162791, 0.02371189, 0.04859666, 0.07332251, 0.08413273, 0.04566858, 0.09413273, 0.11413273, 0.1253278, 0.1523679]

average_pT = data.T[0]
RAA = data1.T[1] / data.T[1]
raa = []
for x, y, z in zip(average_pT, RAA, err): 
    raa.append([x, y, z])

np.savetxt('data/AuAu200_RAA_cut15_coef05', raa)

RAA = data2.T[1] / data.T[1]
raa = []
for x, y, z in zip(average_pT, RAA, err): 
    raa.append([x, y, z])

np.savetxt('data/AuAu200_RAA_cut15_coef1', raa)

RAA = data3.T[1] / data.T[1]
raa = []
for x, y, z in zip(average_pT, RAA, err): 
    raa.append([x, y, z])

np.savetxt('data/AuAu200_RAA_cut15_coef2', raa)

RAA = data4.T[1] / data.T[1]
raa = []
for x, y, z in zip(average_pT, RAA, err): 
    raa.append([x, y, z])

np.savetxt('data/AuAu200_RAA_cut15_coef10', raa)
