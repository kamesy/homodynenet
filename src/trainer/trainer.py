import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch as th
from torchvision.transforms.functional import to_pil_image

from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, TerminateOnNan
from ignite.utils import setup_logger
from ignite.contrib.handlers import ProgressBar

from optim import create_lr_scheduler
from utils.metrics import accuracy


# Ignite Events in order
#   Events.STARTED
#   Events.EPOCH_STARTED
#   Events.GET_BATCH_STARTED
#   Events.GET_BATCH_COMPLETED
#   Events.DATALOADER_STOP_ITERATION
#   Events.ITERATION_STARTED
#   Events.ITERATION_COMPLETED
#   Events.TERMINATE_SINGLE_EPOCH
#   Events.TERMINATE
#   Events.EPOCH_COMPLETED
#   Events.COMPLETED
#   Events.EXCEPTION_RAISED

def create_trainer(loader, model, opt, loss_fn, device, args):

    def _update(engine, batch):
        model.train()

        x = batch['x'].to(engine.state.device, non_blocking=True)
        y = batch['y'].to(engine.state.device, non_blocking=True)
        m = batch['m'].to(engine.state.device, non_blocking=True)

        opt.zero_grad()
        y_pred = model(x)
        loss = loss_fn(m*y_pred, m*y)
        loss.backward()
        opt.step()

        return {
            'x': x.detach(),
            'y': y.detach(),
            'm': m.detach(),
            'y_pred': y_pred.detach(),
            'loss': loss.item()
        }

    def _inference(engine, batch):
        model.eval()

        with th.no_grad():
            x = batch['x'].to(engine.state.device, non_blocking=True)
            y = batch['y'].to(engine.state.device, non_blocking=True)
            m = batch['m'].to(engine.state.device, non_blocking=True)

            y_pred = model(x)
            loss = loss_fn(m*y_pred, m*y)

        return {
            'x': x.detach(),
            'y': y.detach(),
            'm': m.detach(),
            'y_pred': y_pred.detach(),
            'loss': loss.item()
        }

    trainer = Engine(_update)
    evaluator = Engine(_inference)

    logdir = args.logdir
    save_ = not args.nosave

    # initialize trainer state
    trainer.state.device = device
    trainer.state.hparams = args
    trainer.state.save = save_
    trainer.state.logdir = logdir

    trainer.state.df = defaultdict(dict)
    trainer.state.metrics = dict()
    trainer.state.val_metrics = dict()

    # initialize evaluator state
    evaluator.logger = setup_logger('evaluator')
    evaluator.state.device = device
    evaluator.state.save = save_
    evaluator.state.imgdir = logdir + '/images'
    evaluator.state.df = defaultdict(dict)
    evaluator.state.metrics = dict()
    evaluator.state.counter = 0
    evaluator.state.epoch_ = 0

    # TODO: args param
    nval = len(loader["val"])
    save_id = list(range(1, nval, nval // min(nval, 8)))

    if args.tqdm:
        pbar = ProgressBar(persist=True)
        ebar = ProgressBar(persist=False)

        pbar.attach(trainer, ['loss'])
        ebar.attach(evaluator, ['loss'])

    # terminate on nan
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED,
        TerminateOnNan(lambda x: x['loss'])
    )

    # metrics
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED,
        _metrics
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        _metrics_mean
    )

    evaluator.add_event_handler(
        Events.ITERATION_COMPLETED,
        _metrics
    )

    evaluator.add_event_handler(
        Events.COMPLETED,
        _metrics_mean
    )

    trainer.add_event_handler(
        #Events.STARTED | Events.EPOCH_COMPLETED,
        Events.EPOCH_COMPLETED,
        _evaluate, evaluator, loader
    )

    # logging
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        _log_metrics
    )

    # early stopping
    if args.early_stopping > 0:
        es_p = args.early_stopping
        es_s = lambda engine: -engine.state.metrics['loss']
        evaluator.add_event_handler(
            Events.COMPLETED,
            EarlyStopping(patience=es_p, score_function=es_s, trainer=trainer)
        )

    # lr schedulers
    if args.epoch_length is None:
        el = len(loader['train'])
    else:
        el = args.epoch_length

    if args.lr_scheduler is not None:
        lr_sched = create_lr_scheduler(opt, args, num_steps=el)

        if args.lr_scheduler != 'plateau':
            def _sched_fun(engine):
                lr_sched.step()
        else:
            def _sched_fun(engine):
                e = engine.state.epoch
                v = engine.state.val_metrics[e]['nmse']
                lr_sched.step(v)

        if args.lr_scheduler == 'linearcycle' or args.lr_scheduler == 'cycle':
            trainer.add_event_handler(Events.ITERATION_STARTED, lr_sched)
        else:
            trainer.add_event_handler(Events.EPOCH_COMPLETED, _sched_fun)

    # FIXME: warmup is modifying opt base_lr, must create last
    if args.lr_warmup > 0:
        wsched = create_lr_scheduler(opt, args, 'warmup', num_steps=el)
        wsts = wsched.total_steps
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED(event_filter=lambda _, i: i <= wsts),
            lambda _: wsched.step()
        )

    # saving
    if save_:
        to_save = {
            'model': model,
            'optimizer': opt,
            'trainer': trainer,
            'evaluator': evaluator,
        }

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            Checkpoint(to_save, DiskSaver(logdir), n_saved=3)
        )

        evaluator.add_event_handler(
            Events.COMPLETED,
            Checkpoint(
                {'model': model},
                DiskSaver(logdir),
                n_saved = 3,
                filename_prefix = 'best',
                score_function = lambda engine: -1000.0*engine.state.metrics['nmae'],
                score_name = 'val_nmae',
            )
       )

        evaluator.add_event_handler(
            Events.COMPLETED,
            Checkpoint(
                {'model': model},
                DiskSaver(logdir),
                n_saved = 3,
                filename_prefix = 'best',
                score_function = lambda engine: -1000.0*engine.state.metrics['nmse'],
                score_name = 'val_nmse',
            )
        )

        evaluator.add_event_handler(
            Events.COMPLETED,
            Checkpoint(
                {'model': model},
                DiskSaver(logdir),
                n_saved = 3,
                filename_prefix = 'best',
                score_function = lambda engine: 100.0*engine.state.metrics['R2'],
                score_name = 'val_R2',
            )
        )

        if not Path(evaluator.state.imgdir).exists():
            Path(evaluator.state.imgdir).mkdir(parents=True, exist_ok=True)

        evaluator.add_event_handler(
            Events.ITERATION_COMPLETED(event_filter=lambda _, i: i in save_id),
            _save_slices
        )

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            _save_metrics
        )

    return trainer


def _metrics(engine: Engine) -> None:
    i = engine.state.iteration
    e = engine.state.epoch

    l = engine.state.output['loss']
    x = engine.state.output['y_pred']
    y = engine.state.output['y']
    m = engine.state.output['m']
    a = accuracy(m*x, m*y)

    engine.state.df[i]['epoch'] = e
    engine.state.df[i]['loss'] = l
    engine.state.metrics['loss'] = l
    for k, v in a.items():
        engine.state.df[i][k] = v.item()
        engine.state.metrics[k] = v.item()


def _metrics_mean(engine: Engine) -> None:
    e = engine.state.epoch
    df = pd.DataFrame.from_dict(engine.state.df, orient='index')
    df = df[df['epoch'] == e]
    df = df.loc[:, df.columns != 'epoch'].mean()
    d = df.to_dict()
    for k, v in d.items():
        engine.state.metrics[k] = v


def _evaluate(trainer: Engine, evaluator: Engine, loader) -> None:
    e = trainer.state.epoch
    evaluator.state.epoch_ = e
    evaluator.run(loader['val'])
    evaluator.state.counter += 1
    trainer.state.val_metrics[e] = evaluator.state.metrics.copy()


def _log_metrics(engine: Engine) -> None:
    logger = logging.getLogger(__name__)
    e = engine.state.epoch

    for k, v in engine.state.metrics.items():
        logger.info('train_' + '{:<5} : {}'.format(k, v))

    for k, v in engine.state.val_metrics[e].items():
        logger.info('val_' + '{:<7} : {}'.format(k, v))


def _save_metrics(engine: Engine) -> None:
    df = pd.DataFrame.from_dict(engine.state.df, orient='index')
    df.reset_index(inplace=True)
    df.to_feather(engine.state.logdir + '/df.arrow')
    th.save(engine.state.df, engine.state.logdir + '/df.pt')

    df = pd.DataFrame.from_dict(engine.state.val_metrics, orient='index')
    df.reset_index(inplace=True)
    df.to_feather(engine.state.logdir + '/val.arrow')
    th.save(engine.state.val_metrics, engine.state.logdir + '/val.pt')


def _save_slices(engine: Engine) -> None:
    c0, c1 = -0.05, 0.05

    i = engine.state.iteration
    e = engine.state.epoch_
    l = engine.state.output['loss']
    m = engine.state.output['m']
    x = m*engine.state.output['x']
    u = m*engine.state.output['y_pred']
    y = m*engine.state.output['y']

    b, c, h, w = x.size()
    x = x[b//2, 1]
    u = u[b//2, 1]
    y = y[b//2, 1]
    img = th.cat((x, u, y), 1)
    img = th.clamp(img, c0, c1)
    pic = to_pil_image(img.add_(-c0).div_(c1-c0).unsqueeze(0).to('cpu'))
    pic.save(engine.state.imgdir + '/val_{}_epoch_{}_loss_{:.4e}.png'.format(i, e, l))
