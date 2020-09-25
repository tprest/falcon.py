"""
This file implements tests for various parts of the Falcon.py library.

Test the code with:
> make test
"""
from common import q, sqnorm
from fft import add, sub, mul, div, neg, fft, ifft
from ntt import mul_zq, div_zq
from samplerz import samplerz, MAX_SIGMA
from ffsampling import ffldl, ffldl_fft, ffnp, ffnp_fft
from ffsampling import gram
from random import randint, random, gauss, uniform
from math import pi, sqrt, floor, ceil, exp
from ntrugen import karamul, ntru_gen, gs_norm
from falcon import SecretKey, PublicKey, Params, SALT_LEN
from encoding import compress, decompress
from scripts import saga
import sys
if sys.version_info >= (3, 4):
    from importlib import reload  # Python 3.4+ only.


def vecmatmul(t, B):
    """Compute the product t * B, where t is a vector and B is a square matrix.

    Args:
        B: a matrix

    Format: coefficient
    """
    nrows = len(B)
    ncols = len(B[0])
    deg = len(B[0][0])
    assert(len(t) == nrows)
    v = [[0 for k in range(deg)] for j in range(ncols)]
    for j in range(ncols):
        for i in range(nrows):
            v[j] = add(v[j], mul(t[i], B[i][j]))
    return v


def test_fft(n, iterations=10):
    """Test the FFT."""
    for i in range(iterations):
        f = [randint(-3, 4) for j in range(n)]
        g = [randint(-3, 4) for j in range(n)]
        h = mul(f, g)
        k = div(h, f)
        k = [int(round(elt)) for elt in k]
        if k != g:
            print("(f * g) / f =", k)
            print("g =", g)
            print("mismatch")
            return False
    return True


def test_ntt(n, iterations=10):
    """Test the NTT."""
    for i in range(iterations):
        f = [randint(0, q - 1) for j in range(n)]
        g = [randint(0, q - 1) for j in range(n)]
        h = mul_zq(f, g)
        try:
            k = div_zq(h, f)
            if k != g:
                print("(f * g) / f =", k)
                print("g =", g)
                print("mismatch")
                return False
        except ZeroDivisionError:
            continue
    return True


def check_ntru(f, g, F, G):
    """Check that f * G - g * F = 1 mod (x ** n + 1)."""
    a = karamul(f, G)
    b = karamul(g, F)
    c = [a[i] - b[i] for i in range(len(f))]
    return ((c[0] == q) and all(coef == 0 for coef in c[1:]))


def test_ntrugen(n, iterations=10):
    """Test ntru_gen."""
    for i in range(iterations):
        f, g, F, G = ntru_gen(n)
        if check_ntru(f, g, F, G) is False:
            return False
    return True


def test_ffnp(n, iterations):
    """Test ffnp.

    This functions check that:
    1. the two versions (coefficient and FFT embeddings) of ffnp are consistent
    2. ffnp output lattice vectors close to the targets.
    """
    f, g, F, G = ntru_gen(n)
    B = [[g, neg(f)], [G, neg(F)]]
    G0 = gram(B)
    G0_fft = [[fft(elt) for elt in row] for row in G0]
    T = ffldl(G0)
    T_fft = ffldl_fft(G0_fft)

    sqgsnorm = gs_norm(f, g, q)
    m = 0
    for i in range(iterations):
        t = [[random() for i in range(n)], [random() for i in range(n)]]
        t_fft = [fft(elt) for elt in t]

        z = ffnp(t, T)
        z_fft = ffnp_fft(t_fft, T_fft)

        zb = [ifft(elt) for elt in z_fft]
        zb = [[round(coef) for coef in elt] for elt in zb]
        if z != zb:
            print("ffnp and ffnp_fft are not consistent")
            return False
        diff = [sub(t[0], z[0]), sub(t[1], z[1])]
        diffB = vecmatmul(diff, B)
        norm_zmc = int(round(sqnorm(diffB)))
        m = max(m, norm_zmc)
    th_bound = (n / 4.) * sqgsnorm
    if m > th_bound:
        print("Warning: the algorithm does not output vectors as short as expected")
        return False
    else:
        return True


def test_compress(n, iterations):
    """Test compression and decompression."""
    try:
        sigma = 1.5 * sqrt(q)
        slen = Params[n]["sig_bytelen"] - SALT_LEN
    except KeyError:
        return True
    for i in range(iterations):
        while(1):
            initial = [int(round(gauss(0, sigma))) for coef in range(n)]
            compressed = compress(initial, slen)
            if compressed is not False:
                break
        decompressed = decompress(compressed, slen, n)
        # print compressed
        if decompressed != initial:
            return False
    return True


def test_samplerz(nb_mu=100, nb_sig=100, nb_samp=1000):
    """
    Test our Gaussian sampler on a bunch of samples.
    """
    # Minimal size of a bucket for the chi-squared test (must be >= 5)
    chi2_bucket = 10
    # print("Testing the sampler over Z with:\n")
    # print("- {a} different centers\n".format(a=nb_mu))
    # print("- {a} different sigmas\n".format(a=nb_sig))
    # print("- {a} samples per center and sigma\n".format(a=nb_samp))
    assert(nb_samp >= 10 * chi2_bucket)
    q = 12289
    sigmin = 1.3
    nb_rej = 0
    for i in range(nb_mu):
        mu = uniform(0, q)
        for j in range(nb_sig):
            sigma = uniform(sigmin, MAX_SIGMA)
            list_samples = [samplerz(mu, sigma, sigmin) for _ in range(nb_samp)]
            v = saga.UnivariateSamples(mu, sigma, list_samples)
            if (v.is_valid is False):
                nb_rej += 1
    return True
    if (nb_rej > 5 * ceil(saga.pmin * nb_mu * nb_sig)):
        return False
    else:
        return True


def test_falcon(n, iterations=10):
    """Test Falcon."""
    sk = SecretKey(n)
    pk = PublicKey(sk)
    for i in range(iterations):
        message = "0"
        sig = sk.sign(message)
        if pk.verify(message, sig) is False:
            return False
    return True


def test(n, iterations=10):
    """A battery of tests."""
    sys.stdout.write('Test FFT         : ')
    print("OK" if test_fft(n, iterations) else "Not OK")
    sys.stdout.write('Test NTT         : ')
    print("OK" if test_ntt(n, iterations) else "Not OK")
    # test_ntrugen is very costly, hence not performed in higher dimensions
    if (n < 512):
        sys.stdout.write('Test ntru_gen    : ')
        print("OK" if test_ntrugen(n, iterations // 10) else "Not OK")
    sys.stdout.write('Test ffnp        : ')
    print("OK" if test_ffnp(n, iterations) else "Not OK")
    # New test: test_samplerz
    sys.stdout.write('Test samplerz    : ')
    print("OK" if test_samplerz(iterations, iterations, 1000) else "Not OK")
    # test_compress and test_falcon are only performed
    # for parameter sets that are defined.
    if(n in Params):
        sys.stdout.write('Test compression : ')
        print("OK" if test_compress(n, iterations) else "Not OK")
        sys.stdout.write('Test Falcon      : ')
        print("OK" if test_falcon(n, iterations) else "Not OK")


# Run all the tests
if (__name__ == "__main__"):
    for i in range(6, 11):
        n = (1 << i)
        print("Test battery for n = {n}".format(n=n))
        test(n)
