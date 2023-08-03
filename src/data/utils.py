import random

import h5py
import numpy as np
import scipy.io as io
import torch as th


def crop_indices(m):
    if th.is_tensor(m):
        s1 = th.sum(m, -1, keepdim = True)
        s2 = th.sum(m, -2, keepdim = True)

        ix = th.nonzero(th.sum(s2, -3).squeeze(), as_tuple=False).squeeze()
        iy = th.nonzero(th.sum(s1, -3).squeeze(), as_tuple=False).squeeze()
        iz = th.nonzero(th.sum(s1, -2).squeeze(), as_tuple=False).squeeze()

        ix = ix.unsqueeze(0).unsqueeze(1)
        iy = iy.unsqueeze(0).unsqueeze(2)
        iz = iz.unsqueeze(1).unsqueeze(2)

    else:
        s1 = np.sum(m, -1, keepdims = True)
        s2 = np.sum(m, -2, keepdims = True)

        ix = np.nonzero(np.sum(s2, -3).squeeze())[0]
        iy = np.nonzero(np.sum(s1, -3).squeeze())[0]
        iz = np.nonzero(np.sum(s1, -2).squeeze())[0]

        ix = np.expand_dims(ix, axis=(0, 1))
        iy = np.expand_dims(iy, axis=(0, 2))
        iz = np.expand_dims(iz, axis=(1, 2))

    return iz, iy, ix


def _augment(x, hflip=True, vflip=True, rot90=True, invert=True, scale=1):
    scale = float(scale)

    if x is None:
        return x

    if hflip:
        x = x.flip(-1)

    if vflip:
        x = x.flip(-2)

    if rot90:
        x = x.rot90(1, (1, 2))

    if invert and x.dtype in [th.float16, th.float32, th.float64]:
        x *= -1.0

    if scale != 1 and x.dtype in [th.float16, th.float32, th.float64]:
        x *= scale

    return x


def augment(*args, hflip=True, vflip=True, rot=True, invert=True, scale=True):
    kw = dict(
        hflip  = hflip  and random.random() > 0.5,
        vflip  = vflip  and random.random() > 0.5,
        rot90  = rot    and random.random() > 0.5,
        invert = invert and random.random() > 0.5,
        scale  = scale  and random.random() > 0.5 and random.uniform(0.8, 1.2),
    )
    return [_augment(x, **kw) for x in args]


def get_patch(x, *args, patch=96):
    if patch is None or patch <= 0:
        return x, *args

    def _sz(y):
        return tuple(y.shape[-2:])

    szx = _sz(x)

    if szx[0] <= patch and szx[1] <= patch:
        return x, *args

    xi = random.randint(0, szx[1] - patch)
    xj = random.randint(0, szx[0] - patch)

    def fn(y):
        if y is None:
            return y

        szy = _sz(y)

        si = szy[1] // szx[1]
        sj = szy[0] // szx[0]

        pi = si * patch
        pj = sj * patch

        assert pi <= szy[1] and pj <= szy[0]

        yi = si * xi
        yj = sj * xj

        y = y[..., yj:yj+pj, yi:yi+pi]

        return y

    return fn(x), *[fn(y) for y in args]


def load_mat(file, keys):
    keys = [k for k in keys if k]

    try:
        try:
            with h5py.File(str(file), 'r') as h5:
                mat = {k: np.ascontiguousarray(h5[k]) for k in keys}
        except:
            mat = io.loadmat(str(file), variable_names=keys, matlab_compatible=True)
            mat = {k: np.ascontiguousarray(mat[k].T) for k in keys}

    except Exception as e:
        print(file)
        raise e

    return mat
