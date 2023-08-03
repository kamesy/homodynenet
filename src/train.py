import logging
from argparse import ArgumentParser

import torch as th
import torch.backends.cudnn as cudnn

from data import create_dataloader
from loss import create_loss
from models import add_model_args, create_model
from optim import create_optimiser
from trainer import create_trainer
from utils import init_logging, seed_all


def main(args):
    logdir = init_logging(args)
    logger = logging.getLogger(__name__)

    args.logdir = logdir

    if args.cpu or not th.cuda.is_available():
        device = th.device('cpu')
    else:
        device = th.device('cuda')
        cudnn.enabled = True
        cudnn.benchmark = True

    seed_all(args.seed)

    logger.info('Creating dataloader')
    loader = create_dataloader(args)

    logger.info('Creating model')
    model = create_model(args).to(device)

    logger.info('Creating optimiser')
    opt = create_optimiser(model.parameters(), args)

    logger.info('Creating loss')
    loss = create_loss(args)

    logger.info('Creating trainer')
    trainer = create_trainer(loader, model, opt, loss, device, args)

    logger.info('Starting trainer')
    trainer.run(loader['train'], max_epochs=args.epochs, epoch_length=args.epoch_length)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)

    # train/val data arguments
    parser.add_argument(
        '--data',
        required = True,
        type = str,
        help = 'Path to directory containing train/val subdirectories with data'
    )

    parser.add_argument(
        '--data_filter',
        default = None,
        type = str,
        help = 'Regex to filter files in data/{train,val}'
    )

    parser.add_argument(
        '--data_file',
        default = None,
        type = str,
        help = 'Text file in data/{train,val} containing filepaths'
    )

    parser.add_argument(
        '--data_ext',
        default = '.mat',
        choices = ['.mat'],
        type = str,
        help = 'Extension of data files in data/{train,val}'
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

    # logging
    parser.add_argument(
        '--logdir',
        default = './log',
        type = str
    )

    parser.add_argument(
        '--project',
        default = 'homodynenet',
        type = str
    )

    parser.add_argument(
        '--name',
        default = None,
        type = str
    )

    # model
    parser.add_argument(
        '--model',
        default = 'unet',
        choices = ['unet'],
        type = str,
    )

    # model weights
    parser.add_argument(
        '--ckp',
        default = None,
        type = str
    )

    # training
    parser.add_argument(
        '--epochs',
        default = 1000,
        type = int
    )

    parser.add_argument(
        '--epoch_length',
        default = None,
        type = int
    )

    parser.add_argument(
        '--batch_size',
         default = 32,
         type = int
    )

    parser.add_argument(
        '--patch',
        default = 128,
        type = int
    )

    parser.add_argument(
        '--no_augment',
        default = False,
        action = 'store_true'
    )

    parser.add_argument(
        '--num_workers',
        default = 6,
        type = int
    )

    # loss
    parser.add_argument(
        '--loss',
        default = 'l1',
        choices = ['l1', 'l2'],
        type = str
    )

    # optim
    parser.add_argument(
        '--optim',
        default = 'adam',
        choices = ['adagrad', 'adam', 'rmsprop', 'sgd'],
        type = str
    )

    parser.add_argument(
        '--lr',
        default = 1e-4,
        type = float
    )

    # optim.adagrad
    parser.add_argument(
        '--lr_decay',
        default = 0,
        type = float
    )

    # optim.adam
    parser.add_argument(
        '--beta1',
        default = 0.9,
        type = float
    )

    parser.add_argument(
        '--beta2',
        default = 0.999,
        type = float
    )

    # optim.rmsprop
    parser.add_argument(
        '--alpha',
        default = 0.99,
        type = float
    )

    # optim.common
    parser.add_argument(
        '--momentum',
        default = 0.9,
        type = float
    )

    parser.add_argument(
        '--weight_decay',
        default = 0,
        type = float
    )

    # optim.lr_scheduler
    parser.add_argument(
        '--lr_scheduler',
        default = 'plateau',
        choices = ['step', 'multistep', 'exponential', 'plateau', 'linearcycle', 'cycle'],
        type = str
    )

    parser.add_argument(
        '--lr_warmup',
        default = 1,
        type = int
    )

    parser.add_argument(
        '--lr_start',
        default = 1e-8,
        type = float
    )

    # optim.lr_scheduler.StepLR
    parser.add_argument(
        '--step_size',
        default = 30,
        type = int
    )

    # optim.lr_scheduler.MultiStepLR
    parser.add_argument(
        '--milestones',
        default = [30, 60, 90],
        nargs = '+',
        type = int
    )

    # optim.lr_scheduler.ReduceLROnPlateau
    parser.add_argument(
        '--patience',
        default = 8,
        type = int
    )

    parser.add_argument(
        '--patience_factor',
        default = 2,
        type = int
    )

    parser.add_argument(
        '--max_patience',
        default = 64,
        type = int
    )

    parser.add_argument(
        '--min_lr',
        default = 3e-7,
        type = float
    )

    parser.add_argument(
        '--threshold',
        default = 1e-4,
        type = float
    )

    # optim.lr_scheduler.common
    parser.add_argument(
        '--gamma',
        default = 0.25,
        type = float
    )

    parser.add_argument(
        '--early_stopping',
        default = 64,
        type = int
    )

    # misc
    parser.add_argument(
        '--devrun',
        default = False,
        action = 'store_true'
    )

    parser.add_argument(
        '--nosave',
        default = False,
        action = 'store_true'
    )

    parser.add_argument(
        '--cpu',
        default = False,
        action = 'store_true'
    )

    parser.add_argument(
        '--tqdm',
        default = False,
        action = 'store_true'
    )

    parser.add_argument(
        '--seed',
        default = 733,
        type = int
    )

    # TODO
    # parser.add_argument(
    #     '--resume',
    #     default = False,
    #     action = 'store_true'
    # )

    tmp_args, _ = parser.parse_known_args()
    parser = add_model_args(parser, tmp_args)

    args = parser.parse_args()

    if args.devrun:
        args.epoch = 1
        args.epoch_length = 1

    main(args)
