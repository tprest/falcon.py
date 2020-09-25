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
from ntrugen import ntru_gen, gs_norm
from encoding import compress, decompress
# https://pycryptodome.readthedocs.io/en/latest/src/hash/shake256.html
from Crypto.Hash import SHAKE256
import sys
# For debugging purposes
if sys.version_info >= (3, 4):
    from importlib import reload  # Python 3.4+ only.
# Randomness
if sys.version_info < (3, 6):
    from os import urandom as randombytes
else:
    from secrets import token_bytes as randombytes

set_printoptions(linewidth=200, precision=5, suppress=True)


# Bytelength of the signing salt
SALT_LEN = 40
# Max Gram-Schmidt norm of the private basis
MAX_GS_NORM = 129.701241706


# Parameter sets for Falcon:
# - n is the dimension/degree of the cyclotomic ring
# - sigma is the std. dev. of signatures (Gaussians over a lattice)
# - sigmin is a lower bounds on the std. dev. of each Gaussian over Z
# - sigbound is the upper bound on ||s0||^2 + ||s1||^2
# - sig_bytelen is the bytelength of signatures
Params = {
    # FalconParam(128, 32)
    128: {
        "n": 128,
        "sigma": 160.30114421975344,
        "sigmin": 1.235926056771981,
        "sig_bound": 7959734,
        "sig_bytelen": 196,
    },
    # FalconParam(256, 64)
    256: {
        "n": 256,
        "sigma": 163.04153322607107,
        "sigmin": 1.2570545284063217,
        "sig_bound": 16468416,
        "sig_bytelen": 350,
    },
    # FalconParam(512, 128)
    512: {
        "n": 512,
        "sigma": 165.7366171829776,
        "sigmin": 1.2778336969128337,
        "sig_bound": 34034726,
        "sig_bytelen": 666,
    },
    # FalconParam(1024, 256)
    1024: {
        "n": 1024,
        "sigma": 168.38857144654395,
        "sigmin": 1.298280334344292,
        "sig_bound": 70265242,
        "sig_bytelen": 1280,
    },
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


class PublicKey:
    """
    This class contains methods for performing public key operations in Falcon.
    """

    def __init__(self, sk):
        """Initialize a public key."""
        self.n = sk.n
        self.h = sk.h
        self.hash_to_point = sk.hash_to_point
        self.signature_bound = sk.signature_bound
        self.verify = sk.verify

    def __repr__(self):
        """Print the object in readable form."""
        rep = "Public for n = {n}:\n\n".format(n=self.n)
        rep += "h = {h}\n".format(h=self.f)
        return rep


class SecretKey:
    """
    This class contains methods for performing
    secret key operations (and also public key operations) in Falcon.

    One can perform:
    - initializing a secret key for:
        - n = 8, 16, 32, 64, 128, 256, 512, 1024,
        - phi = x ** n + 1,
        - q = 12 * 1024 + 1
    - finding a preimage t of a point c (both in ( Z[x] mod (Phi,q) )**2 ) such that t*B0 = c
    - hashing a message to a point of Z[x] mod (Phi,q)
    - sign a message
    - verify the signature of a message
    """

    def __init__(self, n):
        """Initialize a secret key."""
        # Public parameters
        self.n = n
        self.sigma = Params[n]["sigma"]
        self.sigmin = Params[n]["sigmin"]
        self.signature_bound = Params[n]["sig_bound"]
        self.sig_bytelen = Params[n]["sig_bytelen"]

        # Compute NTRU polynomials f, g, F, G verifying fG - gF = q mod Phi
        while(1):
            self.f, self.g, self.F, self.G = ntru_gen(n)
            sq_gs_norm = gs_norm(self.f, self.g, q)
            if (sq_gs_norm <= MAX_GS_NORM ** 2):
                break

        # Compute fft's of f, g, F, G
        self.f_fft = fft(self.f)
        self.g_fft = fft(self.g)
        self.F_fft = fft(self.F)
        self.G_fft = fft(self.G)

        # From f, g, F, G, compute the basis B0 of a NTRU lattice
        # as well as its Gram matrix and their fft's.
        self.B0 = [[self.g, neg(self.f)], [self.G, neg(self.F)]]
        self.G0 = gram(self.B0)
        self.B0_fft = [[fft(elt) for elt in row] for row in self.B0]
        self.G0_fft = [[fft(elt) for elt in row] for row in self.G0]

        self.T_fft = ffldl_fft(self.G0_fft)

        # Normalize Falcon tree
        # print_tree(self.T_fft)
        normalize_tree(self.T_fft, self.sigma)

        # The public key is a polynomial such that h*f = g mod (Phi,q)
        self.h = div_zq(self.g, self.f)

    def __repr__(self, verbose=False):
        """Print the object in readable form."""
        rep = "Private key for n = {n}:\n\n".format(n=self.n)
        rep += "f = {f}\n".format(f=self.f)
        rep += "g = {g}\n".format(g=self.g)
        rep += "F = {F}\n".format(F=self.F)
        rep += "G = {G}\n".format(G=self.G)
        if verbose:
            rep += "\nFFT tree\n"
            rep += print_tree(self.T_fft, pref="")
        return rep

    def get_coord_in_fft(self, point):
        """Compute t such that t*B0 = c."""
        c0, c1 = point
        [[a, b], [c, d]] = self.B0_fft
        c0_fft, c1_fft = fft(c0), fft(c1)
        t0_fft = [(c0_fft[i] * d[i] - c1_fft[i] * c[i]) / q for i in range(self.n)]
        t1_fft = [(-c0_fft[i] * b[i] + c1_fft[i] * a[i]) / q for i in range(self.n)]
        return t0_fft, t1_fft

    def hash_to_point(self, message, salt):
        """
        Hash a message to a point in Z[x] mod(Phi, q).
        Inspired by the Parse function from NewHope.
        """
        n = self.n
        if q > (1 << 16):
            raise ValueError("The modulus is too large")

        k = (1 << 16) // q
        emessage = message.encode('utf-8')
        shake = SHAKE256.new()
        shake.update(salt)
        shake.update(emessage)
        # digest = hash_instance.hexdigest(8 * n)
        hashed = [0 for i in range(n)]
        i = 0
        j = 0
        while i < n:
            # Takes 2 bytes, transform them in a 16 bits integer
            twobytes = shake.read(2)
            elt = (twobytes[0] << 8) + twobytes[1]  # This breaks in Python 2.x
            # elt = int(digest[4 * j: 4 * (j + 1)], 16)
            # Implicit rejection sampling
            if elt < k * q:
                hashed[i] = elt % q
                i += 1
            j += 1
        return hashed

    def sample_preimage_fft(self, point):
        """Sample preimage."""
        B = self.B0_fft
        c = point, [0] * self.n
        t_fft = self.get_coord_in_fft(c)
        z_fft = ffsampling_fft(t_fft, self.T_fft, self.sigmin)
        v0_fft = add_fft(mul_fft(z_fft[0], B[0][0]), mul_fft(z_fft[1], B[1][0]))
        v1_fft = add_fft(mul_fft(z_fft[0], B[0][1]), mul_fft(z_fft[1], B[1][1]))
        v0 = [int(round(elt)) for elt in ifft(v0_fft)]
        v1 = [int(round(elt)) for elt in ifft(v1_fft)]
        v = v0, v1
        s = [sub(c[0], v[0]), sub(c[1], v[1])]
        return s

    def sign(self, message):
        """Sign a message."""
        salt = randombytes(SALT_LEN)
        hashed = self.hash_to_point(message, salt)
        # We repeat the signing procedure until we find a signature that is
        # short enough (both the Euclidean norm and the bytelength)
        while(1):
            s = self.sample_preimage_fft(hashed)
            norm_sign = sum(coef ** 2 for coef in s[0])
            norm_sign += sum(coef ** 2 for coef in s[1])
            # Check the Euclidean norm
            if norm_sign < self.signature_bound:
                enc_s = compress(s[1], self.sig_bytelen - SALT_LEN)
                # Check that the encoding is valid (sometimes it fails)
                if (enc_s is not False):
                    return salt + enc_s

    def verify(self, message, signature):
        """Verify a signature."""
        salt = signature[:SALT_LEN]
        enc_s = signature[SALT_LEN:]
        s1 = decompress(enc_s, self.sig_bytelen - SALT_LEN, self.n)
        # Check that the encoding is valid
        if (s1 is False):
            print("Invalid encoding")
            return False
        hashed = self.hash_to_point(message, salt)
        s0 = sub_zq(hashed, mul_zq(s1, self.h))
        s0 = [(coef + (q >> 1)) % q - (q >> 1) for coef in s0]
        # Check that the signature is short enough
        norm_sign = sum(coef ** 2 for coef in s0)
        norm_sign += sum(coef ** 2 for coef in s1)
        if norm_sign > self.signature_bound:
            print("Squared norm of signature is too large:", norm_sign)
            return False
        return True
