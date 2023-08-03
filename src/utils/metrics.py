import torch as th

from .utils import norm1, norm2, norm_inf


# l1: absolute errors
def mae(x, y):
    """Mean Absolute Error"""
    return norm1(y - x) / x.numel()


def nmae(x, y):
    """Normalized Mean Absolute Error"""
    return norm1(y - x) / norm1(y)


def mare(x, y):
    """Mean Absolute Relative Error"""
    return norm1((y - x) / y) / x.numel()


def rae(x, y):
    """Relative Absolute Error"""
    return norm1(y - x) / norm1(y - th.mean(y))


def maxae(x, y):
    """Maximum Absolute Error"""
    return norm_inf(y - x)


def absolute_errors(x, y):
    N = x.numel()

    e = y - x
    ae = norm1(e)

    _nmae = ae / norm1(y)
    _rae = ae / norm1(y - th.mean(y))
    _maxae = norm_inf(e)

    return {
        'nmae': _nmae,
        'rae': _rae,
        'maxae': _maxae
    }


# l2: squared errors
def mse(x, y):
    """Mean Squared Error"""
    return norm2(y - x)**2 / x.numel()


def rmse(x, y):
    """Root Mean Squared Error"""
    return th.sqrt(mse(x, y))


def nmse(x, y):
    """Normalized Mean Squared Error"""
    return norm2(y - x)**2 / norm2(y)**2


def msre(x, y):
    """Mean Squared Relative Error"""
    return norm2((y - x) / y)**2 / x.numel()


def Rsquared(x, y):
    """Coefficient of Determination"""
    return 1 - norm2(y - x)**2 / norm2(y - th.mean(y))**2


def psnr(x, y):
    """Peak Signal-to-Noise Ratio"""
    return 10 * th.log10(1 / mse(x, y))


def squared_errors(x, y):
    N = x.numel()

    e = y - x
    se = norm2(e)**2

    _nmse = se / norm2(y)**2
    _nrmse = th.sqrt(_nmse)
    _R2 = 1 - se / norm2(y - th.mean(y))**2

    return {
        'nmse': _nmse,
        'nrmse': _nrmse,
        'R2': _R2,
    }


def accuracy(x, y):
    a = absolute_errors(x, y)
    s = squared_errors(x, y)
    return {**a, **s}
