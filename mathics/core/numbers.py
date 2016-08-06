#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import absolute_import

import sympy
import mpmath
from math import log, ceil
from six.moves import range
import string


C = log(10, 2)  # ~ 3.3219280948873626


# Number of bits of machine precision
machine_precision = 53


def reconstruct_digits(bits):
    '''
    Number of digits needed to reconstruct a number with given bits of precision.

    >>> reconstruct_digits(53)
    17
    '''
    return int(ceil(bits / C) + 1)


class SpecialValueError(Exception):
    def __init__(self, name):
        self.name = name


def get_type(value):
    if isinstance(value, sympy.Integer):
        return 'z'
    elif isinstance(value, sympy.Rational):
        return 'q'
    elif isinstance(value, sympy.Float) or isinstance(value, mpmath.mpf):
        return 'f'
    elif (isinstance(value, sympy.Expr) and value.is_number and
          not value.is_real) or isinstance(value, mpmath.mpc):
        return 'c'
    else:
        return None


def same(v1, v2):
    return get_type(v1) == get_type(v2) and v1 == v2


def dps(prec):
    return max(1, int(round(int(prec) / C - 1)))


def prec(dps):
    return max(1, int(round((int(dps) + 1) * C)))


def min_prec(*args):
    result = None
    for arg in args:
        prec = arg.get_precision()
        if result is None or (prec is not None and prec < result):
            result = prec
    return result


def pickle_mp(value):
    return (get_type(value), str(value))


def unpickle_mp(value):
    type, value = value
    if type == 'z':
        return sympy.Integer(value)
    elif type == 'q':
        return sympy.Rational(value)
    elif type == 'f':
        return sympy.Float(value)
    else:
        return value

# algorithm based on
# http://stackoverflow.com/questions/5110177/how-to-convert-floating-point-number-to-base-3-in-python       # nopep8


def convert_base(x, base, precision=10):
    sign = -1 if x < 0 else 1
    x *= sign

    length_of_int = 0 if x == 0 else int(log(x, base))
    iexps = list(range(length_of_int, -1, -1))
    digits = string.digits + string.ascii_lowercase

    if base > len(digits):
        raise ValueError

    def convert(x, base, exponents):
        out = []
        for e in exponents:
            d = int(x / (base ** e))
            x -= d * (base ** e)
            out.append(digits[d])
            if x == 0 and e < 0:
                break
        return out

    int_part = convert(int(x), base, iexps)
    if sign == -1:
        int_part.insert(0, '-')

    if isinstance(x, (float, sympy.Float)):
        fexps = list(range(-1, -int(precision + 1), -1))
        real_part = convert(x - int(x), base, fexps)

        return "%s.%s" % (''.join(int_part), ''.join(real_part))
    else:
        return ''.join(int_part)


def convert_int_to_digit_list(x, base):
    if x == 0:
        return [0]

    x = abs(x)

    length_of_int = int(log(x, base)) + 1
    iexps = list(range(length_of_int, -1, -1))

    def convert(x, base, exponents):
        out = []
        for e in exponents:
            d = int(x // (base ** e))
            x -= d * (base ** e)
            if out or d != 0:   # drop any leading zeroes
                out.append(d)
            if x == 0 and e < 0:
                break
        return out

    return convert(x, base, iexps)


def round_to_float(expr, evaluation=None, permit_complex=False):
    '''
    Try to round an expression to a python float.
    Return None if not possible.
    '''
    from mathics.core.expression import BaseExpression, Expression, Number
    if not isinstance(expr, BaseExpression):
        raise ValueError(type(expr))
    if evaluation is None:
        value = expr
    else:
        value = Expression('N', expr).evaluate(evaluation)
    if isinstance(value, Number):
        value = value.round()
        return value.get_float_value(permit_complex=permit_complex)
