"""
Python implementation of Falcon:
https://falcon-sign.info/.
"""
from common import q
from numpy import set_printoptions
from math import sqrt
from fft import fft, ifft, sub, neg, add_fft, mul_fft
from ntt import sub_zq, mul_zq, div_zq
from ffsampling import gram, ffldl_fft, ffsampling_fft
from ntrugen import ntru_gen
from encoding import compress, decompress
from dataclasses import dataclass
# https://pycryptodome.readthedocs.io/en/latest/src/hash/shake256.html
from Crypto.Hash import SHAKE256
# Randomness
from os import urandom
from rng import ChaCha20
# For debugging purposes
from beartype import beartype


set_printoptions(linewidth=200, precision=5, suppress=True)

logn = {
    2: 1,
    4: 2,
    8: 3,
    16: 4,
    32: 5,
    64: 6,
    128: 7,
    256: 8,
    512: 9,
    1024: 10
}


# Bytelength of the signing salt and header
HEAD_LEN = 1
SALT_LEN = 40
SEED_LEN = 56


@dataclass
class FalconParam:
    """
    Dataclass for Falcon parameters.
    - n is the dimension/degree of the cyclotomic ring
    - sigma is the std. dev. of signatures (Gaussians over a lattice)
    - sigmin is a lower bounds on the std. dev. of each Gaussian over Z
    - sigbound is the upper bound on ||s0||^2 + ||s1||^2
    - sig_bytelen is the bytelength of signatures
    """
    n: int
    sigma: float
    sigmin: float
    sig_bound: int
    sig_bytelen: int


params = {
    # FalconParam(2)
    2: FalconParam(
        n=2,
        sigma=144.81253976308423,
        sigmin=1.1165085072329104,
        sig_bound=101498,
        sig_bytelen=44,
    ),
    # FalconParam(4)
    4: FalconParam(
        n=4,
        sigma=146.83798833523608,
        sigmin=1.1321247692325274,
        sig_bound=208714,
        sig_bytelen=47,
    ),
    # FalconParam(8)
    8: FalconParam(
        n=8,
        sigma=148.83587593064718,
        sigmin=1.147528535373367,
        sig_bound=428865,
        sig_bytelen=52,
    ),
    # FalconParam(16)
    16: FalconParam(
        n=16,
        sigma=151.78340713845503,
        sigmin=1.170254078853483,
        sig_bound=892039,
        sig_bytelen=63,
    ),
    # FalconParam(32)
    32: FalconParam(
        n=32,
        sigma=154.6747794602761,
        sigmin=1.1925466358390344,
        sig_bound=1852696,
        sig_bytelen=82,
    ),
    # FalconParam(64)
    64: FalconParam(
        n=64,
        sigma=157.51308555044122,
        sigmin=1.2144300507766141,
        sig_bound=3842630,
        sig_bytelen=122,
    ),
    # FalconParam(128)
    128: FalconParam(
        n=128,
        sigma=160.30114421975344,
        sigmin=1.235926056771981,
        sig_bound=7959734,
        sig_bytelen=200,
    ),
    # FalconParam(256)
    256: FalconParam(
        n=256,
        sigma=163.04153322607107,
        sigmin=1.2570545284063217,
        sig_bound=16468416,
        sig_bytelen=356,
    ),
    # FalconParam(512)
    512: FalconParam(
        n=512,
        sigma=165.7366171829776,
        sigmin=1.2778336969128337,
        sig_bound=34034726,
        sig_bytelen=666,
    ),
    # FalconParam(1024)
    1024: FalconParam(
        n=1024,
        sigma=168.38857144654395,
        sigmin=1.298280334344292,
        sig_bound=70265242,
        sig_bytelen=1280,
    )
}


def print_tree(tree, pref=""):
    """
    Display a LDL tree in a readable form.

    Args:
        T: a LDL tree

    Format: coefficient or fft
    """
    leaf = "|_____> "
    top = "|_______"
    son1 = "|       "
    son2 = "        "
    width = len(top)

    a = ""
    if len(tree) == 3:
        if (pref == ""):
            a += pref + str(tree[0]) + "\n"
        else:
            a += pref[:-width] + top + str(tree[0]) + "\n"
        a += print_tree(tree[1], pref + son1)
        a += print_tree(tree[2], pref + son2)
        return a

    else:
        return (pref[:-width] + leaf + str(tree) + "\n")


def normalize_tree(tree, sigma):
    """
    Normalize leaves of a LDL tree (from values ||b_i||**2 to sigma/||b_i||).

    Args:
        T: a LDL tree
        sigma: a standard deviation

    Format: coefficient or fft
    """
    if len(tree) == 3:
        normalize_tree(tree[1], sigma)
        normalize_tree(tree[2], sigma)
    else:
        tree[0] = sigma / sqrt(tree[0].real)
        tree[1] = 0


@beartype
def serialize_poly(poly: list[int]) -> bytes:
    """
    Serialize a polynomial (a list of integers) into a bytestring.
    We assume that all entries are between 0 and q - 1.
    """
    n = len(poly)
    if (min(poly) < 0) or (max(poly) >= q):
        raise ValueError("The entries of poly are outside bounds")

    BITS_PER_COEF = 14
    int_buffer = 0
    for idx in range(n):
        int_buffer ^= poly[idx] << (idx * BITS_PER_COEF)
    # The "+ 7" allows to round to the nearest higher integer.
    bytelen = (n * BITS_PER_COEF + 7) >> 3
    return int_buffer.to_bytes(bytelen)


@beartype
def deserialize_to_poly(bytestring: bytes, n: int) -> list[int]:
    """
    Deserialize a bytestring to a polynomial (a list of integers).
    We assume that all entries are between 0 and q - 1.
    """
    assert (8 * len(bytestring) % n == 0)
    BITS_PER_COEF = 14
    mask = (1 << BITS_PER_COEF) - 1

    poly = [0] * n
    int_buffer = int.from_bytes(bytestring)
    for idx in range(n):
        poly[idx] = int_buffer & mask
        int_buffer >>= BITS_PER_COEF
    return poly


@dataclass
class Falcon:
    """
    A dataclass for Falcon.

    Typical example of how to use this class:

    >>> falcon = Falcon(n).  # n a power-of-two between 2 and 1024
    >>> sk, vk = falcon.keygen()
    >>> sig = falcon.sign(sk, msg)
    >>> assert (falcon.verify(vk, msg, sig) == True)

    """
    param: FalconParam

    @beartype
    def __init__(self, n: int):
        if n in params:
            self.param = params[n]
        else:
            raise ValueError

    def __hash_to_point__(self, message: bytes, salt: bytes) -> list[int]:
        """
        Hash a message to a point in Z[x] mod(Phi, q).
        Inspired by the Parse function from NewHope.
        """
        n = self.param.n
        if q > (1 << 16):
            raise ValueError("The modulus is too large")

        k = (1 << 16) // q
        # Create a SHAKE object and hash the salt and message.
        shake = SHAKE256.new()
        shake.update(salt)
        shake.update(message)
        # Output pseudorandom bytes and map them to coefficients.
        hashed = [0 for i in range(n)]
        i = 0
        while i < n:
            # Takes 2 bytes, transform them in a 16 bits integer
            twobytes = shake.read(2)
            elt = (twobytes[0] << 8) + twobytes[1]  # This breaks in Python 2.x
            # Implicit rejection sampling
            if elt < k * q:
                hashed[i] = elt % q
                i += 1
        return hashed

    @beartype
    def pack_sk(self, sk) -> bytes:
        """
        Pack a Falcon secret key into a bytestring.
        To be used in conjunction with unpack_sk.
        
        Not used in this module but can be useful for exporting keys.
        """
        (f, g, F, G, B0_fft, T_fft) = sk
        sk_bytes = b""
        for poly in (f, g, F, G):
            # We reduce modulo q before serializing
            sk_bytes += serialize_poly([coef % q for coef in poly])
        return sk_bytes

    @beartype
    def unpack_sk(self, sk_bytes: bytes):
        """
        Unpack a bytestring to a Falcon secret key.
        
        How to use:

        >>> falcon = Falcon(n)
        >>> sk, _ = falcon.keygen()
        >>> sk_bytes = falcon.pack(sk)
        >>> sk_2 = falcon.unpack_sk(sk_bytes)
        >>> assert (sk == sk2)
        
        """
        assert (len(sk_bytes) % 4 == 0)
        # There are four polys in sk
        len_poly = len(sk_bytes) // 4

        polys = [None, None, None, None]
        for i in range(4):
            polys[i] = deserialize_to_poly(sk_bytes[i * len_poly:(i + 1) * len_poly], self.param.n)
            polys[i] = [((coef + (q >> 1)) % q) - (q >> 1) for coef in polys[i]]
        sk, _ = self.keygen(polys)
        return sk

    def __sample_preimage__(self, B0_fft, T_fft, point, seed=None):
        """
        Sample a short vector s such that s[0] + s[1] * h = point.
        """
        [[a, b], [c, d]] = B0_fft

        # We compute a vector t_fft such that:
        #     (fft(point), fft(0)) * B0_fft = t_fft
        # Because fft(0) = 0 and the inverse of B has a very specific form,
        # we can do several optimizations.
        point_fft = fft(point)
        t0_fft = [(point_fft[i] * d[i]) / q for i in range(self.param.n)]
        t1_fft = [(-point_fft[i] * b[i]) / q for i in range(self.param.n)]
        t_fft = [t0_fft, t1_fft]

        # We now compute v such that:
        #     v = z * B0 for an integral vector z
        #     v is close to (point, 0)
        if seed is None:
            # If no seed is defined, use urandom as the pseudo-random source.
            z_fft = ffsampling_fft(t_fft, T_fft, self.param.sigmin, urandom)
        else:
            # If a seed is defined, initialize a ChaCha20 PRG
            # that is used to generate pseudo-randomness.
            chacha_prng = ChaCha20(seed)
            z_fft = ffsampling_fft(t_fft, T_fft, self.param.sigmin,
                                   chacha_prng.randombytes)

        v0_fft = add_fft(mul_fft(z_fft[0], a), mul_fft(z_fft[1], c))
        v1_fft = add_fft(mul_fft(z_fft[0], b), mul_fft(z_fft[1], d))
        v0 = [int(round(elt)) for elt in ifft(v0_fft)]
        v1 = [int(round(elt)) for elt in ifft(v1_fft)]

        # The difference s = (point, 0) - v is such that:
        #     s is short
        #     s[0] + s[1] * h = point
        s = [sub(point, v0), neg(v1)]
        return s

    @beartype
    def keygen(self, polys: None|list[list[int]]=None):
        """
        Initialize a secret key.
        """
        # Public parameters
        n = self.param.n

        # Compute NTRU polynomials f, g, F, G verifying fG - gF = q mod Phi
        if polys is None:
            f, g, F, G = ntru_gen(n)
        else:
            assert all((len(poly) == n) for poly in polys)
            [f, g, F, G] = [poly[:] for poly in polys]

        # From f, g, F, G, compute the basis B0 of a NTRU lattice
        # as well as its Gram matrix and their fft's.
        B0 = [[g, neg(f)], [G, neg(F)]]
        G0 = gram(B0)
        B0_fft = [[fft(elt) for elt in row] for row in B0]
        G0_fft = [[fft(elt) for elt in row] for row in G0]

        T_fft = ffldl_fft(G0_fft)

        # Normalize Falcon tree
        normalize_tree(T_fft, self.param.sigma)

        # The public key is a polynomial such that h*f = g mod (Phi,q)
        h = div_zq(g, f)

        sk = (f, g, F, G, B0_fft, T_fft)
        vk = serialize_poly(h)
        return (sk, vk)

    @beartype
    def sign(self, sk, message: bytes, randombytes=urandom) -> bytes:
        """
        Sign a message. The message MUST be a byte string or byte array.
        Optionally, one can select the source of (pseudo-)randomness used
        (default: urandom).
        """
        (f, g, F, G, B0_fft, T_fft) = sk

        int_header = 0x30 + logn[self.param.n]
        header = int_header.to_bytes(1, "little")

        salt = randombytes(SALT_LEN)
        hashed = self.__hash_to_point__(message, salt)

        # We repeat the signing procedure until we find a signature that is
        # short enough (both the Euclidean norm and the bytelength)
        while (1):
            if (randombytes == urandom):
                s = self.__sample_preimage__(B0_fft, T_fft, hashed)
            else:
                seed = randombytes(SEED_LEN)
                s = self.__sample_preimage__(B0_fft, T_fft, hashed, seed=seed)
            norm_sign = sum(coef ** 2 for coef in s[0])
            norm_sign += sum(coef ** 2 for coef in s[1])
            # Check the Euclidean norm
            if norm_sign <= self.param.sig_bound:
                enc_s = compress(s[1], self.param.sig_bytelen - HEAD_LEN - SALT_LEN)
                # Check that the encoding is valid (sometimes it fails)
                if (enc_s is not False):
                    return header + salt + enc_s

    @beartype
    def verify(self, vk: bytes, message: bytes, signature: bytes) -> bool:
        """
        Verify a signature.
        """
        # Unpack vk, the salt and the short polynomial s1
        if (8 * len(vk) % self.param.n != 0):
            raise ValueError(f"This does not seem to be a valid key for Falcon-{self.param.n}")
        h = deserialize_to_poly(vk, self.param.n)
        salt = signature[HEAD_LEN:HEAD_LEN + SALT_LEN]
        enc_s = signature[HEAD_LEN + SALT_LEN:]
        s1 = decompress(enc_s, self.param.sig_bytelen - HEAD_LEN - SALT_LEN, self.param.n)

        # Check that the encoding is valid
        if (s1 is False):
            print("Invalid encoding")
            return False

        # Compute s0 and normalize its coefficients in (-q/2, q/2]
        hashed = self.__hash_to_point__(message, salt)
        s0 = sub_zq(hashed, mul_zq(s1, h))
        s0 = [(coef + (q >> 1)) % q - (q >> 1) for coef in s0]

        # Check that the (s0, s1) is short
        norm_sign = sum(coef ** 2 for coef in s0)
        norm_sign += sum(coef ** 2 for coef in s1)
        if norm_sign > self.param.sig_bound:
            print("Squared norm of signature is too large:", norm_sign)
            return False

        # If all checks are passed, accept
        return True
