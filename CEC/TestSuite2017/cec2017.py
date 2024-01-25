from numpy import ndarray, array, atleast_1d, newaxis, zeros, float64, squeeze
from functools import partial
from os import chdir, getcwd
from typing import Union

from .cec17_test_func import cec17_test_func


def cec17(func_id: int, x) -> Union[float64, ndarray]:
    x: ndarray = atleast_1d(x)
    x: ndarray = array(x, dtype=float64, copy=False, order='C')
    if x.ndim == 1:
        x = x[:, newaxis]
    nx, mx = x.shape
    f: ndarray = zeros(shape=mx, dtype=float64, order='C')

    cwd = getcwd()
    chdir(path='./CEC/TestSuite2017/')
    cec17_test_func(x, f, nx, mx, func_id)
    chdir(path=cwd)

    if f.ndim == 1:
        f = squeeze(f)
    return f


cec17_f1: partial = partial(cec17, 1)
cec17_f2: partial = partial(cec17, 2)
cec17_f3: partial = partial(cec17, 3)
cec17_f4: partial = partial(cec17, 4)
cec17_f5: partial = partial(cec17, 5)
cec17_f6: partial = partial(cec17, 6)
cec17_f7: partial = partial(cec17, 7)
cec17_f8: partial = partial(cec17, 8)
cec17_f9: partial = partial(cec17, 9)
cec17_f10: partial = partial(cec17, 10)
cec17_f11: partial = partial(cec17, 11)
cec17_f12: partial = partial(cec17, 12)
cec17_f13: partial = partial(cec17, 13)
cec17_f14: partial = partial(cec17, 14)
cec17_f15: partial = partial(cec17, 15)
cec17_f16: partial = partial(cec17, 16)
cec17_f17: partial = partial(cec17, 17)
cec17_f18: partial = partial(cec17, 18)
cec17_f19: partial = partial(cec17, 19)
cec17_f20: partial = partial(cec17, 20)
cec17_f21: partial = partial(cec17, 21)
cec17_f22: partial = partial(cec17, 22)
cec17_f23: partial = partial(cec17, 23)
cec17_f24: partial = partial(cec17, 24)
cec17_f25: partial = partial(cec17, 25)
cec17_f26: partial = partial(cec17, 26)
cec17_f27: partial = partial(cec17, 27)
cec17_f28: partial = partial(cec17, 28)
cec17_f29: partial = partial(cec17, 29)
cec17_f30: partial = partial(cec17, 30)
