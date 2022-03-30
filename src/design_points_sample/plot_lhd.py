import matplotlib.pyplot as plt
import numpy as np

def hist_1d_2d(X, Y, nameX, nameY):
    left, width = 0.1, 0.75
    bottom, height = 0.1, 0.75
    spacing = 0.005
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.15]
    rect_histy = [left + width + spacing, bottom, 0.15, height]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes(rect_scatter)
    ax1 = fig.add_axes(rect_histx, sharex=ax)
    ax2 = fig.add_axes(rect_histy, sharey=ax)
    ax1.tick_params(axis="x", labelbottom=False)
    ax2.tick_params(axis="y", labelleft=False)

    ax.scatter(X, Y)
    ax1.hist(X, density=True, rwidth=0.9)
    ax2.hist(Y, orientation='horizontal', density=True, rwidth=0.9)
    ax.set_xlabel(nameX)
    ax.set_ylabel(nameY)

design = np.loadtxt('../../data/lhd_sampling_3d.txt')
i,j = 0,2
ParameterLabels = ['k', '$\\alpha_s^{hard, elas}$', '$\\alpha_s^{hard, inel}$']
hist_1d_2d(design[4:,i], design[4:,j], ParameterLabels[i], ParameterLabels[j])
plt.savefig("../../plots/check_design_13")
