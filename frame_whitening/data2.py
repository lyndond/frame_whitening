import array
import os
import os.path as op
from typing import Optional,Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def load_image(
    data_dir: str = "/mnt/home/tyerxa/ceph/datasets/datasets/vanhateren_imc", 
    crop_size: int = 256, 
    rng: Optional[np.random.Generator] = None,
    ) -> npt.NDArray[np.uint16]:
    """Loads randomly cropped img from van hateren dataset."""

    if rng is None:
        rng = np.random.default_rng()

    files = sorted(os.listdir(data_dir))
    n_images = 10
    rand_idx = rng.choice(range(len(files)), n_images, replace=False)

    filename = files[0]
    with open(op.join(data_dir, filename), 'rb') as handle:
        s = handle.read()
        arr = array.array('H', s)
        arr.byteswap()
    img = np.array(arr, dtype='uint16').reshape(1024, 1536)
    H, W = img.shape

    rand_h = rng.integers(0, H-crop_size, 1)[0]
    rand_w = rng.integers(0, W-crop_size, 1)[0]
    img = img[rand_h:rand_h + crop_size, rand_w:rand_w + crop_size]
    return img


def random_walk(n_steps: int, sigma: float = 1.0) -> Tuple[npt.NDArray[np.int], npt.NDArray[np.int]]:
    """2D Gaussian random walk."""
    x = np.random.normal(0, sigma, n_steps)
    y = np.random.normal(0, sigma, n_steps)
    return np.cumsum(x).astype(int), np.cumsum(y).astype(int)


def get_patches(
    img: npt.NDArray, 
    walk_h: npt.NDArray[np.int], 
    walk_w: npt.NDArray[np.int]
    ) -> List[npt.NDArray]:
    """Get patches within a context."""
    all_images = []
    for di, dj in zip(walk_h, walk_w):
        all_images.append(img[di:di+h, dj:dj+w])
    return all_images


def get_contexts(
    img: npt.NDArray, 
    h: int, 
    w: int, 
    n_contexts: int, 
    sigma: float, 
    n_steps: int, 
    pad_factor: int = 1
    ) -> Tuple[npt.NDArray, npt.NDArray[np.int]]:
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


def plot_context_samples(all_contexts: npt.NDArray, n_samples: int):
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


def plot_patch_stats(all_images: npt.NDArray) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex="all", sharey="all")

    im = ax[0].imshow(np.mean(all_images, 0), cmap="bone")
    plt.colorbar(im)
    ax[0].set(title="Cross-patch mean")

    im = ax[1].imshow(np.var(all_images, 0), cmap="bone")
    ax[1].set(title="Cross-patch variance", xticklabels=[], yticklabels=[])
    plt.colorbar(im)
    fig.tight_layout()

