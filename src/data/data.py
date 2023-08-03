import pathlib
import re

import torch as th
from torch.utils.data import DataLoader, Dataset

from .utils import augment, get_patch, load_mat


class HomodyneData(Dataset):

    def __init__(self, root, keys, patch=None, augment=False, ext='.mat',
                 data_filter=None, data_file=None):
        super(HomodyneData, self).__init__()

        if ext != '.mat':
            raise ValueError('only .mat files are currently supported')

        self.root = pathlib.Path(root)
        self.patch = patch
        self.augment = augment

        # keys in .mat file
        self.ix = keys['x']
        self.iy = keys.get('y', None)
        self.im = keys.get('m', None)

        if self.root.is_file():
            self.files = [self.root]

        elif self.root.is_dir():
            # find all files with extension `ext`
            self.files = sorted(list(self.root.rglob(f'*{ext}')))

            # filter files
            if data_filter is not None:
                p = re.compile(data_filter)
                self.files = [f for f in self.files if p.search(str(f)) is not None]

            # filter some more
            if data_file is not None:
                with open(self.root.joinpath(data_file), 'r') as f:
                    p = set(f.read().splitlines())
                self.files = [f for f in self.files if str(f) in p]

        else:
            raise ValueError(f'{str(root)} is not a file or directory')

    def loader(self, file):
        mat = load_mat(file, (self.ix, self.iy, self.im))
        x = mat[self.ix]
        y = mat.get(self.iy, None)
        m = mat.get(self.im, None)
        return x, y, m

    def transform(self, x, y, m):
        x = th.from_numpy(x)
        y = th.from_numpy(y) if y is not None else None
        m = th.from_numpy(m) if m is not None else None

        if self.patch is not None:
            x, y, m = get_patch(x, y, m, patch=self.patch)

        if self.augment:
            x, y, m = augment(x, y, m)

        # for batching. collate_fn does not like None
        y = y if y is not None else th.empty((0,), dtype=th.float32)
        m = m if m is not None else th.empty((0,), dtype=th.bool)

        return x, y, m

    def __getitem__(self, i):
        file = self.files[i]
        x, y, m = self.loader(file)
        x, y, m = self.transform(x, y, m)
        return {'x': x, 'y': y, 'm': m, 'file': str(file)}

    def __len__(self):
        return len(self.files)


def create_dataloader(args, test=False):
    if test:
        return {
            'test' : _data_loader(args, 'test')
        }
    else:
        return {
            'train': _data_loader(args, 'train'),
            'val': _data_loader(args, 'val')
        }


def _data_loader(args, mode=None):
    root = pathlib.Path(args.data)

    if mode is not None:
        root = root.joinpath(mode)

    if not root.exists():
        raise ValueError(f'{root} does not exist')

    if mode == 'train':
        patch = args.patch
        augment = not args.no_augment
        batch_size = args.batch_size
        shuffle = True
        drop_last = True
    else:
        patch = None
        augment = False
        batch_size = 1
        shuffle = False
        drop_last = False

    return DataLoader(
        dataset = HomodyneData(
            root = root,
            keys = {
                'x': args.data_x,
                'y': args.data_y if hasattr(args, 'data_y') else None,
                'm': args.data_mask if hasattr(args, 'data_mask') else None,
            },
            patch = patch,
            augment = augment,
            ext = args.data_ext,
            data_filter = args.data_filter,
            data_file = args.data_file,
        ),
        batch_size = batch_size,
        num_workers = args.num_workers,
        shuffle = shuffle,
        drop_last = drop_last,
        pin_memory = not args.cpu
    )
