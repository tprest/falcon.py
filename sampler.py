"""This file contains an implementation of a Gaussian sampler over Z."""

from random import randint, uniform        # Generate uniform deviates
from math import exp, floor                # Useful math functions


"""Cumulative distribution table of the Gaussian distribution D of standard
deviation 2, centerered on 0 and discretized over [0,1,2,...,18]. Each
i-th pair of 64-bit integer represents the high and low bits of D[i]."""
half_gaussian_cdt = [[0x55252c92c6309bbb, 0xeff1ff6ad56fd4b1],
                     [0xa0491cdd90d2d8a4, 0x4f0c0f90811e39be],
                     [0xd3edc464091b70d2, 0xcb37ce85d4b8dce9],
                     [0xef924604b256a75c, 0x7a3caaf0cb173bd8],
                     [0xfb1833430a9d94bd, 0x5fbf90ae3141bb81],
                     [0xfed5e6b74505d076, 0x8acc840738ebde7d],
                     [0xffc80bc4696667d7, 0xdcb66b63fa04c293],
                     [0xfff7ba229ae8d348, 0x7a056dafc20b1f65],
                     [0xffff0a0ad5f8437f, 0xd790e30964553636],
                     [0xffffe99c18e415f6, 0x2de7f9836e8afd75],
                     [0xfffffe679eaba03f, 0x6ac67fdd56777058],
                     [0xffffffe9412624c4, 0xfcb045fd8c6b52ab],
                     [0xffffffff02b0abdf, 0x07277f30389a9821],
                     [0xfffffffff7660ff9, 0x026fcb549becf9fc],
                     [0xffffffffffc5ab77, 0xaf28abc08e058ec8],
                     [0xfffffffffffecb84, 0xf1261cbb038fc65b],
                     [0xfffffffffffffb08, 0x2520dda588f6f9f2],
                     [0xfffffffffffffff0, 0x09225e35daae50ab],
                     [0xffffffffffffffff, 0xd852a065f5bc5806]]


"""This is the standard deviation used for the half-Gaussian."""
sigma0 = 2


def sampler_half_gaussian():
    """Sample an integer z according to the half-Gaussian distribution.

    The CDF of the half-Gaussian distribution is given in half_gaussian_cdt.
    """
    u0 = randint(0, (1 << 64) - 1)
    z = 0
    p0 = half_gaussian_cdt[z][0]
    # Scan the CDF table for the index where
    # the uniform deviate x0 exceeds the value of the CDF
    while u0 > p0:
        z += 1
        p0 = half_gaussian_cdt[z][0]
    if u0 < p0:
        return z
    # In the very rare case where u0 = p0, we have to draw additional bits
    if u0 == p0:
        u1 = randint(0, (1 << 64) - 1)
        p1 = half_gaussian_cdt[z][1]
        if u1 < p1:
            return z
        else:
            return z + 1


def sampler_z(sigma, mu):
    """Sample an integer z according to a discrete Gaussian distribution.

    The discrete Gaussian must have:
    - a standard deviation sigma =< sigma0
    - a center mu which may be anywhere in R
    """
    while(1):
        z0 = sampler_half_gaussian()
        b = randint(0, 1)
        z = (2 * b - 1) * z0 + b + floor(mu)
        x = (z0 ** 2) / (2. * (sigma0 ** 2)) - ((z - mu) ** 2) / (2. * (sigma ** 2))
        p = exp(x)
        u = uniform(0, 1)
        if u < p:
            return z
