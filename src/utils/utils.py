import logging
import random
from datetime import datetime
from pathlib import Path

import numpy
import torch as th


def seed_all(seed: int) -> None:
    random.seed(seed)
    th.manual_seed(seed)
    numpy.random.seed(seed)


def norm1(x):
    return th.norm(x, 1)


def norm2(x):
    return th.norm(x, 2)


def norm_inf(x):
    return th.norm(x, float('inf'))


def init_logging(args, filename = 'engine.log'):
    if args.nosave:
        logging.basicConfig(
            level = logging.INFO,
            datefmt = '%Y-%m-%d %H:%M:%S',
            format = '%(asctime)s,%(msecs)03d %(levelname)-8s %(message)s'
        )
        return "/dev/null"

    else:
        date = datetime.now().strftime('%Y%m%d_%H%M%S')

        if args.project is not None:
            date = date + '_' + args.project

        if args.name is not None:
            date = date + '_' + args.name

        logdir = Path(args.logdir).joinpath(date)
        logfile = logdir.joinpath(Path(filename).name)

        if not logdir.exists():
            logdir.mkdir(parents=True, exist_ok=True)

        # file logger
        logging.basicConfig(
            filename = logfile,
            level = logging.INFO,
            datefmt = '%Y-%m-%d %H:%M:%S',
            format = (
                '%(asctime)s,%(msecs)03d '
                '%(pathname)s : %(funcName)s : %(lineno)-4d '
                '%(levelname)-8s %(message)s'
            )
        )

        # console logger
        fmt = logging.Formatter('%(asctime)s,%(msecs)03d %(levelname)-8s %(message)s')

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logging.getLogger().addHandler(ch)

        return logdir.as_posix()
