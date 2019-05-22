import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
import numpy as np


def plot2d(x, y):
    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    axScatter.set_xlim((-lim, lim))
    axScatter.set_ylim((-lim, lim))

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, bins=bins)
    axHisty.hist(y, bins=bins, orientation='horizontal')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    plt.show()


def plot3d(x, y, z, normal_vector):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b']
    ax.scatter(x, y, z, c=[colors[r] for r in normal_vector])
    # for xs, ys, zs, v in x, y, z, normal_vector:
    #     c = colors[v]
    #     ax.scatter(xs, ys, zs, c=c)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def boxplotter(measures, names, title, truth=0):

    fig, ax = plt.subplots()
    pos = np.array(range(len(measures))) + 1
    bp = ax.boxplot(measures, sym='k+', positions=pos,
                    notch=1, bootstrap=5000)
    ax.set_title(title, fontsize=11)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel('estimated entropy [nan]', fontsize=12)
    plt.setp(bp['whiskers'], color='k', linestyle='-')
    plt.setp(bp['fliers'], markersize=3.0)
    if truth is not 0:
        plt.hlines(y=truth, xmin=0 ,xmax=len(names) + 1, linewidth=2, color='r')
    plt.show()

def plot_k_range(k_range, k_values, red_line=0, title='k_range'):
    fig = plt.figure(1, figsize=(15, 9))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    bp = ax.boxplot(k_values)
    plt.setp(bp['whiskers'], color='k', linestyle='-')
    plt.setp(bp['fliers'], markersize=3.0)
    # for k in k_values:
    #     plt.scatter(k_range, k, color='b')
    if red_line is not 0:
        plt.hlines(y=red_line, xmin=1, xmax=k_range[-1], linewidth=2, color='r')
    plt.show()


