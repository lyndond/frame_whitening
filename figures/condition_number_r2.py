import numpy as np
import frame_whitening as fw
import frame_whitening.plot as fwplt

def orth(th: float) -> np.ndarray:
    rot = np.array([[np.cos(th), -np.sin(th)],
                   [np.sin(th), np.cos(th)]])
    return rot


fig, ax = plt.subplots(1,2)
ax[0].axis('square')
ax[0].axis([-1.5,1.5,-1.5,1.5])
W = fw.get_mercedes_frame(parseval=False, jitter=False)
W /= np.linalg.norm(W, axis=0, keepdims=True)

all_thetas = np.arange(0, np.pi, np.pi/48)
kappas = np.ones_like(all_thetas)

all_kappas = np.concatenate((np.arange(1, 10, .01), np.ones(1)*1000))

eps = np.ones(1)*10**-6
for i, th in enumerate(all_thetas):
    V = orth(th)  # rotation matrix

    for k in all_kappas:
        cov = V @ np.diag([k, 1.])*(2/(k+1)) @ V.T
        w = fw.compute_g_opt(R, cov)
        if ((w+eps)>0).all():
            kappas[i] = k
    if np.round(np.rad2deg(th)) in [0., 30., 60., 90., 120., 150.]:
        fwplt.plot_ellipse(V@np.diag([kappas[i],1.])@V.T*(2/(kappas[i]+1)), ax=ax[0])

print(np.rad2deg(all_thetas[np.argmin(kappas)]))

plot2d(R, ax[0])

# [ax[1].scatter(np.rad2deg(all_thetas[i]), kappas[i], color=f'C{i:d}') for i in range(len(all_thetas))];
ax[1].plot(np.rad2deg(all_thetas), kappas, color='k')
ax[1].hlines(1.,0,180, linestyles='--', color='r')
ax[0].set(title='Max condition number elipse', xlabel='x1', ylabel='x2')
ax[1].set(title='Max Condition number', ylabel='condition num', xlabel='degrees',
          yscale='linear', xlim=[0, 180] ,ylim=[.8,9.], xticks=np.linspace(0,180,7))
fig.tight_layout()
