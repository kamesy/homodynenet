import logging

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from utils.scheduler import PlateauScheduler


class LinearLR(_LRScheduler):

    def __init__(self, optimizer, max_lr, total_steps, last_epoch=-1):
        self.max_lr = max_lr
        self.total_steps = total_steps
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        else:
            return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        if self.last_epoch > self.total_steps - 1:
            return [self.max_lr for _ in self.base_lrs]
        else:
            b = self.max_lr
            i = self.last_epoch
            n = self.total_steps
            return [a + (b - a) * i / (n - 1) for a in self.base_lrs]


# same as torch.optim.ReduceLROnPlateau v1.6 with modified logging
class ReduceLROnPlateau1(PlateauScheduler):

    def __init__(self, optimizer, min_lr, factor=0.1, eps=1e-8, verbose=True, **kwargs):
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.verbose = verbose
        self.eps = eps
        super().__init__(**kwargs)
        self._reset()

    def _reset(self):
        super()._reset()

    def step(self, metrics):
        ret = super().step(metrics)
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        return ret

    def _fn(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    logger = logging.getLogger(__name__)
                    logger.info('LR scheduler: new learning rate of group'
                                ' {} is {:.4e}.'.format(i, new_lr))

        if self.patience_factor > 0:
            old_p = self.patience
            new_p = min(old_p * self.patience_factor, self.max_patience)
            self.patience = new_p

            if self.verbose and new_p != old_p:
                logger = logging.getLogger(__name__)
                logger.info('LR scheduler: new patience is {:3d}'.format(new_p))

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
