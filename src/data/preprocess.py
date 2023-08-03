import pathlib
import re
import shutil

from argparse import ArgumentParser

import numpy as np
import pandas as pd
import scipy.io as io

from utils import crop_indices, load_mat


def split_dataset(files, train_split):
    """
    Randomly split dataset into train/val/test sets.

    Arguments:
        files: list of homodyne filtered data
        train_split: fraction of data to use for training, val/test are (1-train_split)/2
    """
    if not files:
        raise ValueError('no files')

    if train_split >= 1 or train_split <= 0:
        raise ValueError(f'train split must be in (0, 1), got {train_split}')

    # /path/to/data/filename_hann_n_128_stuff_e1.mat -> _hann_n_128_stuff_e1.mat
    p = re.compile(r'_(?:hann|hamming|gaussian|fermi|unwrap)_.*?_[ew]\d\.mat$', re.IGNORECASE)

    # get unique filenames
    s = set()
    for file in files:
        f = str(file)
        s.add(f[:p.search(f).start()])

    hfiles = sorted([*s])

    # randomize order
    rng = np.random.default_rng(42)
    rng.shuffle(hfiles)

    i1 = int(train_split * len(hfiles))
    i2 = int((1 + train_split) / 2 * len(hfiles))

    # split into train/val/test
    train = []
    val   = []
    test  = []

    for file in files:
        f = str(file)
        for (i, hfile) in enumerate(hfiles):
            if f.startswith(hfile):
                if i < i1:
                    train.append(file)
                elif i < i2:
                    val.append(file)
                else:
                    test.append(file)
                break

    return train, val, test


def preprocess(files, root, outdir, vtol, othr, mode):
    """
    Preprocess homodyne filtered data.

    Split a list with paths to 3D homodyne filtered data into train/val/test
    sets. For the test set the 3D data is copied to the output directory. For
    the train and val sets, 2D slices are extracted from the 3D data if the
    ratio of nonzero voxels to background voxels is greater than the voxel
    tolerance. The slices and their two neighboring slices are saved as a `.mat`
    file in the output directory. The mean and variance of each homodyne filtered
    data as well as the R2 value between the homodyne filtered local field
    (hfl1) and the processed homodyne filtered phase (hfl2) are saved in an
    arrow (`summary.arrow`) file in the output directory.
    Outliers for the second step (hfl2) are removed using Tukey's fences on
    the previously computed R2 values. The filenames of the good slices are
    stored in a text file (`data_tukey_{threshold}.txt`) in the output
    directory.

    Arguments:
        files: list of `.mat` files with homodyne filtered data with keys
            `hfl1`, `hfl2`, `fl`, and `mask`
        root: root directory of homodyne filtered data
        outdir: output directory
        vtol: voxel tolerances for train/val slices
        othr: multiplier of IQR for Tukey's fences
        mode: `train`/`val`/`test`
    """
    ext = '.mat'

    # create output directory
    outpath = pathlib.Path(outdir).joinpath(mode)
    outpath.mkdir(parents=True, exist_ok=True)

    # resolve relative paths
    root = str(root.resolve())
    outpath = str(outpath.resolve())

    # copy test files
    if mode == 'test':
        for (i, src) in enumerate(files):
            if i == 0 or i % 100 == 0 or i == len(files)-1:
                print('[{:5d}/{}] Copying {}'.format(i+1, len(files), src))
            dest = pathlib.Path(str(src).replace(root, outpath))
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        return  # void

    # process train/val data
    stats = {}

    for (i, file) in enumerate(files):
        if i == 0 or (i+1) % 100 == 0 or i == len(files)-1:
            print('[{:5d}/{}] Processing {}'.format(i+1, len(files), file))

        dat  = load_mat(file, ('hfl1', 'hfl2', 'fl', 'mask'))
        mask = dat['mask'].astype(np.bool_)
        hfl1 = dat['hfl1'].astype(np.float32)
        hfl2 = dat['hfl2'].astype(np.float32)
        fl   = dat['fl'].astype(np.float32)

        # crop data
        iz, iy, ix = crop_indices(mask)
        mask = mask[iz, iy, ix]
        hfl1 = hfl1[iz, iy, ix]
        hfl2 = hfl2[iz, iy, ix]
        fl   = fl[iz, iy, ix]

        nz_half = mask.shape[0] // 2

        # create parent directories
        dest = pathlib.Path(str(file).replace(root, outpath))
        dest.parent.mkdir(parents=True, exist_ok=True)

        dest = str(dest).replace(ext, '')

        # save slices
        for k in range(1, mask.shape[0]-1):
            tol = vtol[0] if k < nz_half else vtol[1]
            m = mask[k,:,:]

            if m.sum() > tol * m.size:
                filek = f'{dest}_slice_{k+1}{ext}'
                maskk = mask[k-1:k+2,:,:]
                hfl1k = maskk * hfl1[k-1:k+2,:,:]
                hfl2k = maskk * hfl2[k-1:k+2,:,:]
                flk   = maskk * fl[k-1:k+2,:,:]

                stats[filek] = {
                    'mean_hfl1': np.mean(hfl1k),
                    'var_hfl1': np.var(hfl1k),

                    'mean_hfl2': np.mean(hfl2k),
                    'var_hfl2': np.var(hfl2k),

                    'mean_fl': np.mean(flk),
                    'var_fl': np.var(flk),

                    'R2': 1 - np.linalg.norm(hfl1k - hfl2k)**2 /
                        np.linalg.norm(hfl1k - np.mean(hfl1k))**2,
                }

                io.savemat(filek, {
                    'mask': np.transpose(maskk, (2,1,0)),
                    'hfl1': np.transpose(hfl1k, (2,1,0)),
                    'hfl2': np.transpose(hfl2k, (2,1,0)),
                    'fl': np.transpose(flk, (2,1,0)),
                }, do_compression=True)

    # save stats as dataframe
    outpath = pathlib.Path(outpath)
    spath = outpath.joinpath('summary.arrow')
    print(f'Saving summary to {spath}')

    df = pd.DataFrame.from_dict(stats, orient='index')
    df.reset_index(inplace=True)
    df.to_feather(spath)

    # remove outliers
    q1 = df['R2'].quantile(0.25)
    q3 = df['R2'].quantile(0.75)
    iqr = q3 - q1
    lb = q1 - othr*iqr
    ub = q3 + othr*iqr

    T = df.loc[lb <= df['R2']]

    print('Removing slices with artifacts...')
    print('Tukey\'s fences: [{:.4f}, {:.4f}]'.format(lb, ub))
    print('Data R2 min: {:.4f}, mean: {:.4f}'.format(df['R2'].min(), df['R2'].mean()))
    print('Removed {} slices'.format(len(df) - len(T)))

    tpath = outpath.joinpath(f'data_tukey_{othr}.txt')
    print(f'Saving good slices to {tpath}')

    with open(tpath, 'w') as f:
        for index, row in T.iterrows():
            f.write(row['index'] + '\n')


def main(args):
    indir = args.data
    outdir = args.outdir
    ext = args.ext if args.ext[0] == '.' else '.' + args.ext
    train_split = args.train_split
    vtol = args.voxel_cutoff
    othr = args.outlier_threshold

    if len(vtol) == 1:
        vtol = [vtol[0], vtol[0]]

    root = pathlib.Path(indir)
    files = list(root.rglob(f'*{ext}'))

    train, val, test = split_dataset(files, train_split)
    preprocess(train, root, outdir, vtol, othr, 'train')
    preprocess(val, root, outdir, vtol, othr, 'val')
    preprocess(test, root, outdir, vtol, othr, 'test')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        '--data',
        required = True,
        type = str,
        help = 'Path to directory containing 3D homodyne data'
    )

    parser.add_argument(
        '--outdir',
        required = True,
        type = str,
        help = 'Path to directory where to save preprocessed data'
    )

    parser.add_argument(
        '--train_split',
        default = 0.9,
        type = float,
        help = 'Fraction of data to use for training; val/test are \
            (1-train_split)/2. Default = 0.9'
    )

    parser.add_argument(
        '--voxel_cutoff',
        default = [1/3, 1/4],
        nargs = '+',
        type = float,
        help = 'Ratios of voxels that must be nonzero [top half, bottom half], \
            for a slice to be included in train/val. Default = [1/3, 1/4]'
    )

    parser.add_argument(
        '--outlier_threshold',
        default = 1.5,
        type = float,
        help = 'Multiplier of IQR for Tukey\'s fences for outlier removal. \
            Default = 1.5'
    )

    parser.add_argument(
        '--ext',
        default = '.mat',
        choices = ['.mat'],
        type = str
    )

    args = parser.parse_args()
    main(args)
