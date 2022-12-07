import os
import os.path as op
import array
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import torch
from pyrtools.tools.convolutions import blurDn

### TRANSFORMS
class ToFloatTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # turn to Size([C=1, H, W]) float tensor
        image = torch.as_tensor(sample).float().view(1, *sample.shape)
        return image

class ToNumpy(object):
    """Convert Tensor in sample to ndarray."""

    def __call__(self, sample):
        # turn to Size([H, W]) numpy array
        return sample.squeeze().numpy()

class Whiten(object):
    """Applies Olshausen and Field Style Whitening to an Image"""

    def __init__(self, size):
        super().__init__()
        self.size = size
        self.whiten_mat = self.init_whiten_mat(self.size)

    def __call__(self, img):
       # center and scale img
       mu = img.mean()
       std = img.std()
       img = (img - mu) / std

       # fourier transform
       im_fft = np.fft.fft2(img) 
       im_fft = np.fft.fftshift(im_fft)

       # spectral whtien
       whitened = self.whiten_mat * im_fft

       # invert
       img_whitened = np.fft.ifft2(np.fft.ifftshift(whitened))

       return img_whitened

    def init_whiten_mat(self, crop_size):
        # cutoff frequency 
        f_0 = 0.4 * crop_size
        
        # linear ramp (inverts 1/f)
        r_x = np.arange(-1.*crop_size/2., crop_size/2., 1.)
        r_y = np.arange(-1.*crop_size/2., crop_size/2., 1.)

        f_x, f_y = np.meshgrid(r_x, r_y)

        # filter is linear ramp * lowpass at cutoff frequency
        rho = (f_x**2 + f_y**2)**0.5
        filt = rho * np.exp(-1.*(rho/f_0)**4)

        return filt

class Standardize(object):
    """Standardize image to have given mean and variance"""

    def __init__(self, mu, sigma):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def __call__(self, img):
        img_0_1 = (img - img.mean()) / img.std()

        return (img_0_1 * self.sigma) + self.mu


### DATASET AND DATALOADER
class VanHaterenDataset(Dataset):
    def __init__(self,
                 imc=True,
                 whiten=True,
                 n_blur=0,
                 transform: torchvision.transforms = None,
                 n_ims: int = 100):

        # choose dataset
        if imc:
            root_dir = '/mnt/home/tyerxa/ceph/datasets/datasets/vanhateren_imc'
        else:
            root_dir = '/mnt/home/tyerxa/ceph/datasets/datasets/vanhateren_iml'

        self.data_dir = op.join(root_dir)
        self.filenames = tuple(sorted(os.listdir(root_dir)))
        self.filenames = self.filenames[:n_ims]
        self.transform = transform
        self.all_idx = sorted([int(s[3:8]) for s in self.filenames])
        self.n_ims = n_ims
        self.whiten = whiten
        if self.whiten:
            self.Whitener = Whiten(1024)
        self.n_blur = n_blur

        self.loaded_dict = {}

    def _load_image(self, idx):

        filename = self.filenames[idx]
        img_idx = int(filename[3:8])

        if idx in self.loaded_dict.keys():
            return self.loaded_dict[idx], img_idx

        with open(op.join(self.data_dir, filename), 'rb') as handle:
            s = handle.read()
            arr = array.array('H', s)
            arr.byteswap()
        img = np.array(arr, dtype='int16').reshape(1024, 1536)
        img = img[:, :1024] # taking this to make images square

        for blur_step in range(self.n_blur):
            blurDn(img)

        if self.whiten:
            img = self.Whitener(img)

        for blur_step in range(self.n_blur):
            img = blurDn(img)

        self.loaded_dict[idx] = img

        return img, img_idx

    def __len__(self):
        return self.n_ims

    def __getitem__(self, idx):
        img, idx = self._load_image(idx)
        return self.transform(img), idx


class VanHaterenPatchDataLoader(DataLoader):
    def __init__(
        self,
        transform=None,
        whiten=True,
        n_blur=0,
        imc=True,
        **kwargs
    ):
        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([ToFloatTensor() 
                ])

        dataset = VanHaterenDataset(whiten=whiten, n_blur=n_blur, imc=imc, transform=self.transform)

        super().__init__(dataset, shuffle=False, **kwargs)


if __name__ == '__main__':
    loader = VanHaterenPatchDataLoader(batch_size=2, shuffle=False)
