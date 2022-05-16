import matplotlib.pyplot as plt
import numpy as np
import frame_whitening as fw
from frame_whitening import plot as fwplt
# Used this fig to demonstrate the method.

C, _ = fw.randcov2()
W = fw.get_mercedes_frame()
fig, ax = plt.subplots(1,1)
fwplt.plot_ellipse(C, ax=ax, color='darkgrey')
ax.hlines(0, -2, 2)
ax.vlines(0, -2, 2)

ax.axis('square')
ax.axis((-2,2,-2,2))
ax.set(xlabel='x1', ylabel='x2', xticks=[], yticks=[])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

x = np.linspace(-2,2,10)
for i in range(3):
    ax.plot(x, (W[1,i]/W[0,i])*x, '--', color=f'C{i:d}')


