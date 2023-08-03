import torch as th


def add_model_args(parser, args):
    Model = _get_model(args.model)
    return Model.add_model_args(parser)


def create_model(args):
    Model = _get_model(args.model)
    m = Model(**vars(args))

    if args.ckp is not None:
        ckp = th.load(args.ckp)
        m.load_state_dict(ckp)

    return m


def _get_model(name):
    name = name.lower()
    if name == 'unet':
        from .unet import UNet as m
    else:
        raise ValueError(f'unknown model: {name}')
    return m
