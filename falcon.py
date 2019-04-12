"""Reference implementation of Falcon: https://falcon-sign.info/."""

from common import q
from numpy import set_printoptions
from math import sqrt
from fft import fft, ifft, sub, neg, add_fft, mul_fft
from ntt import add_zq, mul_zq, div_zq
from ffsampling import gram, ffldl_fft, ffsampling_fft
from ntrugen import ntru_gen, gs_norm
from random import randint
from encoding import compress, decompress

# If Python has version >= 3.6, then the built-in hashlib has shake_256.
# Otherwise, sha3 has to be loaded to monkey-patch hashlib.
# See https://pypi.python.org/pypi/pysha3.
import sys
import hashlib
if sys.version_info < (3, 6):
    import sha3

if sys.version_info >= (3, 4):
    from importlib import reload  # Python 3.4+ only.

set_printoptions(linewidth=200, precision=5, suppress=True)


def infinity_max(vector):
    return max(max(abs(x.real), abs(x.imag)) for x in vector)


def infinity_max_tree(tree):
    if len(tree) == 3:
        max_0 = infinity_max(tree[0])
        max_1 = infinity_max_tree(tree[1])
        max_2 = infinity_max_tree(tree[2])
        return max(max_0, max_1, max_2)

    else:
        return infinity_max(tree)


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
        # max_0 = infinity_max(tree[0])
        if (pref == ""):
            a += pref + str(tree[0]) + "\n"
        else:
            a += pref[:-width] + top + str(tree[0]) + "\n"
        a += print_tree(tree[1], pref + son1)
        a += print_tree(tree[2], pref + son2)
        return a
        # return max(max_0, max_1, max_2)

    else:
        # max_t = infinity_max(tree)
        return (pref[:-width] + leaf + str(tree) + "\n")
        # return max_t


def normalize_tree(tree, sigma):
    """
    Normalize the leaves of a LDL tree (from values ||b_i||**2 to sigma/||b_i||).

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
    """This class constains methods for performing public key operations in Falcon."""

    def __init__(self, sk):
        """Docstring."""
        self.n = sk.n
        self.q = sk.q
        self.h = sk.h
        self.hash_to_point = sk.hash_to_point
        self.signature_bound = sk.signature_bound
        self.verify = sk.verify


class SecretKey:
    """
    This class contains methods for performing secret key operations (and also public key operations) in Falcon.

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
        """Public parameters"""
        self.n = n
        self.q = q
        self.hash_function = hashlib.shake_256

        """Private key part 1: NTRU polynomials f, g, F, G verifying fG - gF = q mod Phi"""
        self.f, self.g, self.F, self.G = ntru_gen(n)

        """Private key part 2: fft's of f, g, F, G"""
        self.f_fft = fft(self.f)
        self.g_fft = fft(self.g)
        self.F_fft = fft(self.F)
        self.G_fft = fft(self.G)

        """Private key part 3: from f, g, F, G, compute the basis B0 of a NTRU lattice as well as its Gram matrix and their fft's"""
        self.B0 = [[self.g, neg(self.f)], [self.G, neg(self.F)]]
        self.G0 = gram(self.B0)
        self.B0_fft = [[fft(elt) for elt in row] for row in self.B0]
        self.G0_fft = [[fft(elt) for elt in row] for row in self.G0]

        # self.T = ffldl(self.G0)
        self.T_fft = ffldl_fft(self.G0_fft)

        """Private key part 4: compute sigma and signature bound."""
        slack = 1.1
        smooth = 1.28
        sq_gs_norm = gs_norm(self.f, self.g, q)
        self.sigma = smooth * sqrt(sq_gs_norm)
        self.signature_bound = slack * 2 * self.n * (self.sigma**2)

        """Private key part 5: set leaves of tree to be the standard deviations."""
        print_tree(self.T_fft)
        normalize_tree(self.T_fft, self.sigma)

        """Public key: h such that h*f = g mod (Phi,q)"""
        self.h = div_zq(self.g, self.f)

    def __repr__(self, verbose=True):
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
        t0_fft = [(c0_fft[i] * d[i] - c1_fft[i] * c[i]) / self.q for i in range(self.n)]
        t1_fft = [(-c0_fft[i] * b[i] + c1_fft[i] * a[i]) / self.q for i in range(self.n)]
        return t0_fft, t1_fft

    def hash_to_point(self, message, salt):
        """Hash a message to a point in Z[x] mod(Phi, q).

        Inspired by the Parse function from NewHope.
        """
        n, q = self.n, self.q
        if q > 2 ** 16:
            raise ValueError("The modulus is too large")

        k = (2 ** 16) / q
        # We take twice the number of bits that would be needed if there was no rejection
        emessage = message.encode('utf-8')
        esalt = salt.encode('utf-8')
        hash_instance = self.hash_function()
        hash_instance.update(esalt)
        hash_instance.update(emessage)
        digest = hash_instance.hexdigest(8 * n)
        hashed = [0 for i in range(n)]
        i = 0
        j = 0
        while i < n:
            # Takes 2 bytes, transform them in a 16 bits integer
            elt = int(digest[4 * j: 4 * (j + 1)], 16)
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
        # print("t0,", infinity_max(t_fft[0]))
        # print("t1,", infinity_max(t_fft[1]))
        z_fft = ffsampling_fft(t_fft, self.T_fft)
        # print("z0,", infinity_max(z_fft[0]))
        # print("z1,", infinity_max(z_fft[1]))
        # print("t_fft = {t_fft}".format(t_fft=t_fft))
        # print("z_fft = {z_fft}".format(z_fft=z_fft))
        v0_fft = add_fft(mul_fft(z_fft[0], B[0][0]), mul_fft(z_fft[1], B[1][0]))
        v1_fft = add_fft(mul_fft(z_fft[0], B[0][1]), mul_fft(z_fft[1], B[1][1]))
        v0 = [int(round(elt)) for elt in ifft(v0_fft)]
        v1 = [int(round(elt)) for elt in ifft(v1_fft)]
        v = v0, v1
        s = [sub(c[0], v[0]), sub(c[1], v[1])]
        # print("s0,", infinity_max(s[0]))
        # print("s1,", infinity_max(s[1]))
        return s

    def sign(self, message, salt=None, err=0):
        """Sign a message. Needs hash randomization to be secure."""
        """1. The message is hashed into a point of Z[x] mod (Phi,q)."""
        if salt is None:
            salt = randint(0, (1 << 320) - 1)
        r = ""
        for i in range(320 // 8):
            r += chr((salt >> (8 * i)) & 0xff)
        hashed = self.hash_to_point(message, r)
        """2. A short pre-image of this point is determined."""
        while(1):
            s = self.sample_preimage_fft(hashed)
            """3. The norm of the signature is checked."""
            norm_sign = sum(sum(elt**2 for elt in part) for part in s)
            if norm_sign < self.signature_bound:
                return r, s
            else:
                print("redo")

    def verify(self, message, signature):
        """Verify a signature."""
        r, s = signature
        """1. hashes a message to a point of Z[x] mod (Phi,q"""
        hashed = self.hash_to_point(message, r)
        """2. Computes s0 + s1*h."""
        result = add_zq(s[0], mul_zq(s[1], self.h))
        """3. Verifies that the s0 + s1*h = hashed."""
        if any(result[i] != hashed[i] for i in range(self.n)):
            print("The signature does not correspond to the hash!")
            return False
        """4. Verifies that the norm is small"""
        norm_sign = sum(sum(elt**2 for elt in part) for part in s)
        # print "signature bound   = ", self.signature_bound
        # print "norm of signature = ", norm_sign
        if norm_sign > self.signature_bound:
            print("The squared norm of the signature is too big:", norm_sign)
            return False
        """5. If the previous steps did not fail, accept."""
        return True
