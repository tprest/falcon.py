# Importing dependencies
from math import floor
import sys

# If possible, use high-quality randomness
if sys.version_info >= (3, 6):
    from secrets import randbits
else:
    from random import randint

    def randbits(k):
        return randint(0, (1 << k) - 1)


# Upper bound on all the values of sigma
# INV_2SIGMA2 = 1 / (2 * (MAX_SIGMA ** 2))
MAX_SIGMA = 1.8205
INV_2SIGMA2 = 0.15086504887537272

# Precision of RCDT
RCDT_PREC = 72

# ln(2) and 1 / ln(2), with ln the natural logarithm
LN2 = 0.69314718056
ILN2 = 1.44269504089


# RCDT is the reverse cumulative distribution table of a distribution that
# is very close to a half-Gaussian of parameter MAX_SIGMA.
RCDT = [
    3024686241123004913666,
    1564742784480091954050,
    636254429462080897535,
    199560484645026482916,
    47667343854657281903,
    8595902006365044063,
    163297957344668388,
    117656387352093658,
    8867391802663976,
    496969357462633,
    20680885154299,
    638331848991,
    14602316184,
    247426747,
    3104126,
    28824,
    198,
    1]


# C contains the coefficients of a polynomial that approximates exp(-x)
# More precisely, the value:
# (2 ** -63) * sum(C[12 - i] * (x ** i) for i in range(i))
# Should be very close to exp(-x).
# This polynomial is lifted from FACCT: https://doi.org/10.1109/TC.2019.2940949
C = [
    0x00000004741183A3,
    0x00000036548CFC06,
    0x0000024FDCBF140A,
    0x0000171D939DE045,
    0x0000D00CF58F6F84,
    0x000680681CF796E3,
    0x002D82D8305B0FEA,
    0x011111110E066FD0,
    0x0555555555070F00,
    0x155555555581FF00,
    0x400000000002B400,
    0x7FFFFFFFFFFF4800,
    0x8000000000000000]


def basesampler(source=randbits):
    """
    Sample z0 in {0, 1, ..., 18} with a distribution
    very close to the half-Gaussian D_{Z+, 0, MAX_SIGMA}.
    Takes as (optional) input the randomness source (default: randbits).
    """
    u = source(RCDT_PREC)
    z0 = 0
    for elt in RCDT:
        z0 += int(u < elt)
    return z0


def approxexp(x, ccs):
    """
    Compute an approximation of 2^63 * ccs * exp(-x).

    Input:
    - a floating-point number x
    - a scaling factor ccs
    Both inputs x and ccs MUST be positive.

    Output:
    - an integral approximation of 2^63 * ccs * exp(-x).
    """
    # y, z are always positive
    y = C[0]
    # Since z is positive, int is equivalent to floor
    z = int(x * (1 << 63))
    for elt in C[1:]:
        y = elt - ((z * y) >> 63)
    z = int(ccs * (1 << 63)) << 1
    y = (z * y) >> 63
    return y


def berexp(x, ccs, source=randbits):
    """
    Return a single bit, equal to 1 with probability ~ ccs * exp(-x).
    Both inputs x and ccs MUST be positive.
    Also takes as (optional) input the randomness source (default: randbits).
    """
    s = int(x * ILN2)
    r = x - s * LN2
    s = min(s, 63)
    z = (approxexp(r, ccs) - 1) >> s
    for i in range(56, -8, -8):
        w = source(8) - ((z >> i) & 0xFF)
        if w:
            break
    return (w < 0)


def samplerz(mu, sigma, sigmin, source=randbits):
    """
    Given floating-point values mu, sigma (and sigmin),
    output an integer z according to the discrete
    Gaussian distribution D_{Z, mu, sigma}.

    Input:
    - the center mu
    - the standard deviation sigma
    - a scaling factor sigmin
    - optional: the randomness source (default: randbits)
    The inputs MUST verify 1 < sigmin < sigma < MAX_SIGMA.

    Output:
    - a sample z from the distribution D_{Z, mu, sigma}.
    """
    # assert(sigma < MAX_SIGMA), sigma
    # assert(sigmin < sigma)
    # assert(1 < sigmin)
    s = int(floor(mu))
    r = mu - s
    dss = 1 / (2 * sigma * sigma)
    ccs = sigmin / sigma
    while(1):
        # Sampler z0 from a Half-Gaussian
        z0 = basesampler(source=source)
        # Convert z0 into a pseudo-Gaussian sample z
        b = source(8) & 1
        z = b + (2 * b - 1) * z0
        # Rejection sampling to obtain a true Gaussian sample
        x = ((z - r) ** 2) * dss
        x -= (z0 ** 2) * INV_2SIGMA2
        if berexp(x, ccs, source=source):
            return z + s
