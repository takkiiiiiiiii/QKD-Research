import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.style.use('default')
plt.rcParams["font.family"] = "Times New Roman"

# plt.rcParams["text.usetex"] = True
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8

plt.rcParams['legend.fontsize'] = 13
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['figure.titlesize'] = 20

plt.rcParams['lines.markeredgewidth'] = 1
plt.rcParams['grid.linestyle'] = '-.'
plt.rcParams['legend.facecolor'] = 'white'
plt.rcParams['legend.framealpha'] = 1

twinx_color_1 = '#d95319'
twinx_color_2 = '#5eb8f2'


def loadMarker():
    return itertools.cycle(('o', 'v', '^', 's', 'p', 'D', '*'))


def loadLineStyle():
    return itertools.cycle(('-', 'dotted', '--', '-.'))


def loadColor():
    return itertools.cycle(('r', 'g', 'b', 'm', 'c'))