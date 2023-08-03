def create_optimiser(params, args):
    name = args.optim.lower()
    params = (p for p in params if p.requires_grad)

    lr = args.lr
    wd = args.weight_decay

    if name == 'adam':
        b = [args.beta1, args.beta2]
        if wd > 0:
            from torch.optim import AdamW
            opt = AdamW(params, lr, betas=b, weight_decay=wd)
        else:
            from torch.optim import Adam
            opt = Adam(params, lr, betas=b)

    elif name == 'sgd':
        from torch.optim import SGD
        m = args.momentum
        opt = SGD(params, lr, momentum=m, nesterov=True, weight_decay=wd)

    elif name == 'rmsprop':
        from torch.optim import RMSprop
        m = args.momentum
        a = args.alpha
        opt = RMSprop(params, lr, momentum=m, alpha=a, weight_decay=wd)

    elif name == 'adagrad':
        from torch.optim import Adagrad
        ld = args.lr_decay
        opt = Adagrad(params, lr, lr_decay=ld, weight_decay=wd)

    else:
        raise ValueError('optim must be one of adam, sgd, rmsprop, adagrad')

    return opt


def create_lr_scheduler(opt, args, name = None, num_steps=1):
    if name is None:
        name = args.lr_scheduler.lower()

    g = args.gamma

    if name == 'plateau':
        from .lr_scheduler import ReduceLROnPlateau1
        p = args.patience
        pf = args.patience_factor
        mp = args.max_patience
        ml = args.min_lr
        t = args.threshold
        sched = ReduceLROnPlateau1(
            opt,
            factor = g,
            patience = p,
            patience_factor = pf,
            max_patience = mp,
            min_lr = ml,
            threshold = t,
            verbose = True
        )

    elif name == 'warmup':
        from .lr_scheduler import LinearLR
        for param_group in opt.param_groups:
            param_group['lr'] = args.lr_start
        n = int(num_steps * args.lr_warmup)
        sched = LinearLR(opt, args.lr, n)

    elif name == 'step':
        from torch.optim.lr_scheduler import StepLR
        s = args.step_size
        sched = StepLR(opt, step_size=s, gamma=g)

    elif name == 'multistep':
        from torch.optim.lr_scheduler import MultiStepLR
        m = args.milestones
        sched = MultiStepLR(opt, milestones=m, gamma=g)

    elif name == 'exponential':
        from torch.optim.lr_scheduler import ExponentialLR
        sched = ExponentialLR(opt, gamma=g)

    elif name == 'linearcycle':
        from ignite.contrib.handlers import LinearCyclicalScheduler
        n = int(num_steps * args.epochs / 8)
        sched = LinearCyclicalScheduler(opt, 'lr', args.lr_start, args.lr, n, end_value_mult=0.5)

    elif name == 'cycle':
        from ignite.contrib.handlers import LinearCyclicalScheduler, ConcatScheduler, CosineAnnealingScheduler
        N = int(num_steps * args.epochs//2)
        n = N // 4
        sched1 = LinearCyclicalScheduler(opt, 'lr', args.lr_start, args.lr, n)
        sched2 = CosineAnnealingScheduler(opt, 'lr', args.lr, args.gamma*args.lr,
                                          cycle_size = n,
                                          end_value_mult=args.gamma,
                                          start_value_mult=3*args.gamma/2)
        sched = ConcatScheduler(
            schedulers = [sched1, sched2],
            durations = [n//2, ]
        )

    else:
        raise ValueError(
            'lr_scheduler must be one of plateau, step, multistep, exponential, '
            'linearcycle'
        )

    return sched
