import matplotlib.pyplot as plt
import numpy as np


def random_walk(n_steps, sigma=1.0):
    # 2d gaussian random walk
    x = np.random.normal(0, sigma, n_steps)
    y = np.random.normal(0, sigma, n_steps)
    return np.cumsum(x).astype(int), np.cumsum(y).astype(int)


def get_patches(img, walk_h, walk_w):
    # get patches within a context
    all_images = []
    for di, dj in zip(walk_h, walk_w):
        all_images.append(img[di:di+h, dj:dj+w])
    return all_images


def get_contexts(img, h, w, n_contexts, sigma, n_steps, pad_factor=1):
    pad_h, pad_w = pad_factor * h, pad_factor * w

    all_contexts = []
    walk_coords = []

    for _ in range(n_contexts):
        i, j = np.random.randint(pad_h, img_h-pad_w), np.random.randint(pad_h, img_w-pad_w)
        walk_h, walk_w = random_walk(n_steps, sigma)
        walk_h = np.clip(walk_h+i, 0+h, img_h-h)
        walk_w = np.clip(walk_w+j, 0+w, img_w-w)
        all_contexts.append(get_patches(img, walk_h, walk_w))
        walk_coords.append(np.stack([walk_h, walk_w], axis=1))

    return np.array(all_contexts), walk_coords


# sample 5 images without replacement from all_images and plot
def add_subplot_border(ax, width=1, color=None ):
    """from https://stackoverflow.com/questions/45441909/how-to-add-a-fixed-width-border-to-subplot"""

    fig = ax.get_figure()

    # Convert bottom-left and top-right to display coordinates
    x0, y0 = ax.transAxes.transform((0, 0))
    x1, y1 = ax.transAxes.transform((1, 1))

    # Convert back to Axes coordinates
    x0, y0 = ax.transAxes.inverted().transform((x0, y0))
    x1, y1 = ax.transAxes.inverted().transform((x1, y1))

    rect = plt.Rectangle(
        (x0, y0), x1-x0, y1-y0,
        color=color,
        transform=ax.transAxes,
        zorder=-1,
        lw=2*width+1,
        fill=None,
    )
    fig.patches.append(rect)


def plot_context_samples(all_contexts, n_samples):
    n_contexts = all_contexts.shape[0]
    sampled_idx = np.random.choice(n_steps, n_samples, replace=False)
    fig, ax = plt.subplots(n_contexts, n_samples, figsize=(n_samples, 4))
    for ctx in range(n_contexts):
        VMIN, VMAX = np.min(all_contexts[ctx]), np.max(all_contexts[ctx])
        for i in range(n_samples):
            ax[ctx, i].imshow(all_contexts[ctx][sampled_idx[i]], cmap="bone", vmin=VMIN, vmax=VMAX)
            add_subplot_border(ax[ctx, i], width=3, color=cols[ctx])
            ax[ctx, i].axis("off")

    fig.tight_layout()


def plot_patch_stats(all_images):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex="all", sharey="all")

    im = ax[0].imshow(np.mean(all_images, 0), cmap="bone")
    plt.colorbar(im)
    ax[0].set(title="Cross-patch mean")

    im = ax[1].imshow(np.var(all_images, 0), cmap="bone")
    ax[1].set(title="Cross-patch variance", xticklabels=[], yticklabels=[])
    plt.colorbar(im)
    fig.tight_layout()
