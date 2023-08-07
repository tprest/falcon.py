"""This file contains an implementation of the NTT.

The NTT implemented here is for polynomials in Z_q[x]/(phi), with:
- The integer modulus q = 12 * 1024 + 1 = 12289
- The polynomial modulus phi = x ** n + 1, with n a power of two, n =< 1024

The code is voluntarily very similar to the code of the FFT.
It is probably possible to use templating to merge both implementations.
"""
from common import split, merge, q                     # Import split and merge
from ntt_constants import roots_dict_Zq, inv_mod_q     # Import constants useful for the FFT


"""i2 is the inverse of 2 mod q."""
i2 = 6145


""" sqr1 is a square root of (-1) mod q (currently, sqr1 = 1479)."""
sqr1 = roots_dict_Zq[2][0]


def split_ntt(f_ntt):
    """Split a polynomial f in two or three polynomials.

    Args:
        f_ntt: a polynomial

    Format: NTT
    """
    n = len(f_ntt)
    w = roots_dict_Zq[n]
    f0_ntt = [0] * (n // 2)
    f1_ntt = [0] * (n // 2)
    for i in range(n // 2):
        f0_ntt[i] = (i2 * (f_ntt[2 * i] + f_ntt[2 * i + 1])) % q
        f1_ntt[i] = (i2 * (f_ntt[2 * i] - f_ntt[2 * i + 1]) * inv_mod_q[w[2 * i]]) % q
    return [f0_ntt, f1_ntt]


def merge_ntt(f_list_ntt):
    """Merge two or three polynomials into a single polynomial f.

    Args:
        f_list_ntt: a list of polynomials

    Format: NTT
    """
    f0_ntt, f1_ntt = f_list_ntt
    n = 2 * len(f0_ntt)
    w = roots_dict_Zq[n]
    f_ntt = [0] * n
    for i in range(n // 2):
        f_ntt[2 * i + 0] = (f0_ntt[i] + w[2 * i] * f1_ntt[i]) % q
        f_ntt[2 * i + 1] = (f0_ntt[i] - w[2 * i] * f1_ntt[i]) % q
    return f_ntt


def ntt(f):
    """Compute the NTT of a polynomial.

    Args:
        f: a polynomial

    Format: input as coefficients, output as NTT
    """
    n = len(f)
    if (n > 2):
        f0, f1 = split(f)
        f0_ntt = ntt(f0)
        f1_ntt = ntt(f1)
        f_ntt = merge_ntt([f0_ntt, f1_ntt])
    elif (n == 2):
        f_ntt = [0] * n
        f_ntt[0] = (f[0] + sqr1 * f[1]) % q
        f_ntt[1] = (f[0] - sqr1 * f[1]) % q
    return f_ntt


def intt(f_ntt):
    """Compute the inverse NTT of a polynomial.

    Args:
        f_ntt: a NTT of a polynomial

    Format: input as NTT, output as coefficients
    """
    n = len(f_ntt)
    if (n > 2):
        f0_ntt, f1_ntt = split_ntt(f_ntt)
        f0 = intt(f0_ntt)
        f1 = intt(f1_ntt)
        f = merge([f0, f1])
    elif (n == 2):
        f = [0] * n
        f[0] = (i2 * (f_ntt[0] + f_ntt[1])) % q
        f[1] = (i2 * inv_mod_q[sqr1] * (f_ntt[0] - f_ntt[1])) % q
    return f


def add_zq(f, g):
    """Addition of two polynomials (coefficient representation)."""
    assert len(f) == len(g)
    deg = len(f)
    return [(f[i] + g[i]) % q for i in range(deg)]


def neg_zq(f):
    """Negation of a polynomials (any representation)."""
    deg = len(f)
    return [(- f[i]) % q for i in range(deg)]


def sub_zq(f, g):
    """Substraction of two polynomials (any representation)."""
    return add_zq(f, neg_zq(g))


def mul_zq(f, g):
    """Multiplication of two polynomials (coefficient representation)."""
    return intt(mul_ntt(ntt(f), ntt(g)))


def div_zq(f, g):
    """Division of two polynomials (coefficient representation)."""
    try:
        return intt(div_ntt(ntt(f), ntt(g)))
    except ZeroDivisionError:
        raise


# def adj(f):
#     """Ajoint of a polynomial (coefficient representation)."""
#     return intt(adj_ntt(ntt(f)))


def add_ntt(f_ntt, g_ntt):
    """Addition of two polynomials (NTT representation)."""
    return add_zq(f_ntt, g_ntt)


def sub_ntt(f_ntt, g_ntt):
    """Substraction of two polynomials (NTT representation)."""
    return sub_zq(f_ntt, g_ntt)


def mul_ntt(f_ntt, g_ntt):
    """Multiplication of two polynomials (coefficient representation)."""
    assert len(f_ntt) == len(g_ntt)
    deg = len(f_ntt)
    return [(f_ntt[i] * g_ntt[i]) % q for i in range(deg)]


def div_ntt(f_ntt, g_ntt):
    """Division of two polynomials (NTT representation)."""
    assert len(f_ntt) == len(g_ntt)
    deg = len(f_ntt)
    if any(elt == 0 for elt in g_ntt):
        raise ZeroDivisionError
    return [(f_ntt[i] * inv_mod_q[g_ntt[i]]) % q for i in range(deg)]


# def adj_ntt(f_ntt):
#     """Ajoint of a polynomial (NTT representation)."""
#     deg = len(f_ntt)
#     return [f_ntt[i].conjugate() for i in range(deg)]


"""This value is the ratio between:
    - The degree n
    - The number of complex coefficients of the NTT
While here this ratio is 1, it is possible to develop a short NTT such that it is 2.
"""
ntt_ratio = 1
