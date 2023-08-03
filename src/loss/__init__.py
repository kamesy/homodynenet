def create_loss(args):
    name = args.loss.lower()

    if name == 'l1':
        from torch.nn import L1Loss
        loss = L1Loss(reduction='sum')

    elif name == 'l2':
        from torch.nn import MSELoss
        loss = MSELoss(reduction='sum')

    else:
        raise ValueError('loss must be one of l1, l2')

    return loss
