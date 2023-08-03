import numpy as np
import torch as th
import torch.nn as nn


_ND = 2


def convnd(ch, k, stride=1, padding=None, padding_mode='zeros', dilation=1,
           groups=1, bias=False):
    m = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[_ND]
    return m(ch[0], ch[1],
             kernel_size = k,
             stride = stride,
             padding = (k-1)//2,
             padding_mode = padding_mode,
             dilation = dilation,
             groups = groups,
             bias = bias)


def convnd_t(ch, k, stride=1, padding=None, padding_mode='zeros', dilation=1,
             groups=1, bias=False):
    m = {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}[_ND]
    return m(ch[0], ch[1],
             kernel_size = k,
             stride = stride,
             padding = (k-1)//2,
             padding_mode = padding_mode,
             dilation = dilation,
             groups = groups,
             bias = bias)


class DownBlock(nn.Module):

    def __init__(self, mode='max', k=2, stride=2, ch=None, act=None,
                 akwargs=None, nrm=None, nkwargs=None, bias=False):
        super().__init__()
        mode = mode.lower()

        akwargs = akwargs or {}
        nkwargs = nkwargs or {}
        nrm = nrm or 'id'
        act = act or 'id'

        if ch is not None and ch[0] != ch[1]:
            raise ValueError('input, output channels must be equal')

        if mode == 'max':
            m = {1: nn.MaxPool1d, 2: nn.MaxPool2d, 3: nn.MaxPool3d}[_ND]
            self.down = m(k, stride=stride)

        elif mode == 'mean':
            m = {1: nn.AvgPool1d, 2: nn.AvgPool2d, 3: nn.AvgPool3d}[_ND]
            self.down = m(k, stride=stride)

        elif mode == 'norm2':
            m = {1: nn.LPPool1d, 2: nn.LPPool2d}[_ND]
            self.down = nn.LPPool2d(2, k, stride=stride)

        elif mode == 'conv':
            self.down = convnd(ch, k=k, stride=stride, bias=bias)

        elif mode == 'convna':
            self.down = nn.Sequential(
                convnd(ch, k=k, stride=stride, bias=False),
                normlayer(nrm, ch[1], **nkwargs),
                actlayer(act, inplace=True, **akwargs)
            )

        else:
            raise ValueError("mode must be one of maxpool, norm2, conv(na)")

    def forward(self, x):
        return self.down(x)


class UpBlock(nn.Module):

    def __init__(self, mode='nearest', k=2, stride=2, ch=None, act=None,
                 akwargs=None, nrm=None, nkwargs=None, bias=False):
        super().__init__()
        mode.lower()

        akwargs = akwargs or {}
        nkwargs = nkwargs or {}
        nrm = nrm or 'id'
        act = act or 'id'

        if ch is not None and ch[0] != ch[1]:
            raise ValueError('input, output channels must be equal')

        if mode == 'conv':
            self.up = convnd_t(ch, k=k, stride=stride, bias=bias)

        elif mode == 'convna':
            self.up = nn.Sequential(
                convnd_t(ch, k=k, stride=stride, bias=False),
                normlayer(nrm, ch[1], **nkwargs),
                actlayer(act, inplace=True, **akwargs)
            )

        else:
            self.up = nn.Upsample(scale_factor=stride, mode=mode)

    def forward(self, x):
        return self.up(x)


class BasicBlock(nn.Module):

    def __init__(self, ch, k=3, act='prelu', akwargs=None, skip=False,
                 nrm=None, nkwargs=None, bias=False, depth=0, L=1, alpha=1):
        super().__init__()
        akwargs = akwargs or {}
        nkwargs = nkwargs or {}
        nrm = nrm or 'id'

        ch1, ch2 = ch

        if ch1 == ch2:
            self.bottle = nn.Identity()
        else:
            self.bottle = convnd((ch1, ch2), k=1, bias=bias)

        self.conv1 = convnd((ch2, ch2), k, bias=bias)
        self.norm1 = normlayer(nrm, ch2, **nkwargs)
        self.act1 = actlayer(act, inplace=True, **akwargs)

        self.conv2 = convnd((ch2, ch2), k, bias=bias)
        self.norm2 = normlayer(nrm, ch2, **nkwargs)
        self.act2 = actlayer(act, inplace=True, **akwargs)

        self.skip = skip

        # init
        mo = 'fan_out'
        nl = 'leaky_relu' if act == 'prelu' else 'relu'

        if ch1 != ch2:
            _nl = 'conv2d' if ch2 > ch1 else nl
            nn.init.kaiming_uniform_(self.bottle.weight, mode=mo, nonlinearity=_nl)

        nn.init.kaiming_uniform_(self.conv1.weight, mode=mo, nonlinearity=nl)
        nn.init.kaiming_uniform_(self.conv2.weight, mode=mo, nonlinearity=nl)
        with th.no_grad():
            self.conv1.weight /= np.sqrt(L)
            self.conv2.weight /= np.sqrt(L)
            if isinstance(self.bottle, nn.Conv2d):
                self.bottle.weight /= np.sqrt(depth+1)

    def forward(self, x):
        out = self.bottle(x)
        identity = out

        out = self.conv1(out)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)

        if self.skip:
            out += identity

        return out


_ACTLAYERS = dict(
    relu = nn.ReLU,
    relu6 = nn.ReLU6,
    selu = nn.SELU,
    hardsigmoid = nn.Hardsigmoid,
    hardswish = nn.Hardswish,
    elu = nn.ELU,
    celu = nn.CELU,
    leakyrelu = nn.LeakyReLU,
    rrelu = nn.RReLU,
    prelu = nn.PReLU,
    id = nn.Identity,
    identity = nn.Identity,
)

def actlayer(name, inplace: bool = False, **kwargs):
    name = name.lower()
    if not name in _ACTLAYERS:
        raise ValueError(f"unknown activation: {name}")

    m = _ACTLAYERS[name]

    ks = []
    kw = dict(inplace=inplace)

    if name == 'elu' or name == 'celu':
        ks = ['alpha']

    elif name == 'leakyrelu':
        ks = ['negative_slope']

    elif name == 'rrelu':
        ks = ['lower', 'upper']

    elif name == 'prelu':
        kw = dict() # no-inplace
        ks = ['num_parameters', 'init']

    for k in ks:
        if k in kwargs:
            kw[k] = kwargs[k]

    return m(**kw)


_NORMLAYERS = dict(
    batch1 = nn.BatchNorm1d,
    batch2 = nn.BatchNorm2d,
    batch3 = nn.BatchNorm3d,
    instance1 = nn.InstanceNorm1d,
    instance2 = nn.InstanceNorm2d,
    instance3 = nn.InstanceNorm3d,
    group = nn.GroupNorm,
    layer = nn.LayerNorm,
    id = nn.Identity,
    identity = nn.Identity,
)

def normlayer(name, ch, **kwargs):
    name = name.lower()

    if name == 'batch' or name == 'instance':
        name += str(_ND)

    if not name in _NORMLAYERS:
        raise ValueError(f"unknown norm layer: {name}")

    m = _NORMLAYERS[name]

    ks = []
    kw = dict()

    if 'batch' in name  or 'instance' in name:
        kw['num_features'] = ch
        ks = ['eps', 'momentum', 'affine', 'track_running_stats']

    elif name == 'group':
        kw['num_channels'] = ch
        ks = ['num_groups', 'eps', 'affine']

    elif name == 'layer':
        ks = ['normalized_shape', 'eps', 'elementwise_affine']

    for k in ks:
        if k in kwargs:
            kw[k] = kwargs[k]

    return m(**kw)
