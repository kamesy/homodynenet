import pathlib
from argparse import ArgumentParser

import pandas as pd
import scipy.io as io
import torch as th
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import crop_indices, HomodyneData
from models import add_model_args, create_model
from utils.metrics import accuracy


def main(args):
    if args.cpu or not th.cuda.is_available():
        device = th.device('cpu')
    else:
        print('Initializing gpu')
        device = th.device('cuda')
        cudnn.enabled = True
        cudnn.benchmark = True

    # data loader
    ext = '.mat'
    keyx = args.data_x
    keyy = args.data_y
    keym = args.data_mask

    loader = DataLoader(
        dataset = HomodyneData(
            root = args.data,
            keys = {'x': keyx, 'y': keyy, 'm': keym},
            ext  = ext,
        ),
        num_workers = args.num_workers,
        batch_size = None,
        pin_memory = not args.cpu,
    )

    # prepare output directory
    root = pathlib.Path(args.data)

    if root.is_file():
        root = root.parent

    if args.outdir is not None:
        outdir = pathlib.Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        outdir = root

    print('Loading model')
    model = create_model(args).to(device)
    model.eval()

    # evaluate
    stats = {}

    for batch in tqdm(loader) if args.tqdm else loader:
        with th.no_grad():
            x = batch['x'].to(device, non_blocking=True).squeeze(0)
            Y = batch['y'].to(device, non_blocking=True).squeeze(0)
            m = batch['m'].to(device, non_blocking=True).squeeze(0)

            iz, iy, ix = crop_indices(m)
            mc = m[iz, iy, ix]
            xc = x[iz, iy, ix]
            Yc = Y[iz, iy, ix]

            y0 = th.zeros(xc.shape).to(device)
            y1 = th.zeros(xc.shape).to(device)
            y2 = th.zeros(xc.shape).to(device)

            # homodynenet
            n = xc.size()[0]
            for k in range(1, n-1):
                u = xc[k-1:k+2,...].unsqueeze(0)
                u = model(u)
                y0[k-1,...] = u[0,0,...]
                y1[k,...]   = u[0,1,...]
                y2[k+1,...] = u[0,2,...]

            y1[ 0,...] = y0[ 0,...]
            y1[-1,...] = y2[-1,...]
            yc = (y0 + y1 + y2).div_(3)

            # compute metric
            yc *= mc
            Yc *= mc
            stats[batch['file']] = {k: v.item() for k, v in accuracy(yc, Yc).items()}

            # save
            if args.save_results:
                outpath = batch['file'].replace(ext, f'_{args.suffix}{ext}')

                if args.outdir is not None:
                    outpath = pathlib.Path(outpath.replace(str(root), str(outdir)))
                    outpath.parent.mkdir(parents=True, exist_ok=True)
                    outpath = str(outpath)

                y = th.zeros(x.shape)
                y[iz, iy, ix] = yc.to('cpu')

                io.savemat(outpath, {
                    keyx: y.permute(2, 1, 0).numpy(),
                }, do_compression=True)

    # save stats as dataframe
    ckp = pathlib.Path(args.ckp)
    outpath = outdir.joinpath(f'{ckp.parent.stem}_{ckp.stem}_evaluate.arrow')
    print(f'Saving stats to {outpath}')

    df = pd.DataFrame.from_dict(stats, orient='index')
    df.reset_index(inplace=True)
    df.to_feather(outpath)

    print(df.describe())


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        '--ckp',
        required = True,
        type = str,
        help = 'Path to model checkpoint, can be single file or directory'
    )

    parser.add_argument(
        '--data',
        required = True,
        type = str,
        help = 'Path to homodyne filtered data, can be single file or directory'
    )

    parser.add_argument(
        '--outdir',
        type = str,
        help = 'Output directory. Default = `data`'
    )

    parser.add_argument(
        '--save_results',
        default = False,
        action = 'store_true',
        help = 'Save results to .mat files'
    )

    parser.add_argument(
        '--suffix',
        default = 'homodynenet',
        type = str,
        help = 'Suffix appended to filename. Default = `homodynenet`'
    )

    parser.add_argument(
        '--data_filter',
        default = None,
        type = str,
        help = 'Regex to filter files in data'
    )

    parser.add_argument(
        '--data_file',
        default = None,
        type = str,
        help = 'Text file in data containing filepaths'
    )

    parser.add_argument(
        '--data_ext',
        default = '.mat',
        choices = ['.mat'],
        type = str,
        help = 'Extension of data files in data'
    )

    parser.add_argument(
        '--data_x',
        default = 'hfl1',
        type = str,
        help = 'Name of filtered data, ie input to model, in .mat file'
    )

    parser.add_argument(
        '--data_y',
        default = 'fl',
        type = str,
        help = 'Name of unfiltered data, ie ground truth, in .mat file'
    )

    parser.add_argument(
        '--data_mask',
        default = 'mask',
        type = str,
        help = 'Name of mask in .mat file'
    )

    parser.add_argument(
        '--cpu',
        default = False,
        action = 'store_true'
    )

    parser.add_argument(
        '--num_workers',
        default = 6,
        type = int,
    )

    parser.add_argument(
        '--tqdm',
        default = False,
        action = 'store_true'
    )

    parser.add_argument(
        '--model',
        default = 'unet',
        choices = ['unet'],
        type = str,
    )

    tmp_args, _ = parser.parse_known_args()
    parser = add_model_args(parser, tmp_args)

    args = parser.parse_args()
    main(args)
