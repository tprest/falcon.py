"""References: [DP16] https://eprint.iacr.org/2015/1014."""

# from numpy.random import randint
from common import q
from numpy import set_printoptions
from math import sqrt
from sha3 import shake_256
# from sampler import sampler_z
from fft import fft, ifft, sub, neg
from ntt import add_zq, mul_zq, div_zq
from ffsampling import gram, ffldl, ffldl_fft, ffnp_fft, checknorm, vecmatmul
from ntrugen import ntru_gen
from random import randint


import sys
set_printoptions(linewidth=200, precision=10, suppress=True)


def print_tree(tree, pref="       "):
	"""
	Display a LDL tree in a readable form.

	Args:
		T: a LDL tree

	Format: coefficient or fft
	"""
	leaf = "\_____>"
	top = "\______"
	son1 = "|      "
	son2 = "       "
	width = len(top)

	if len(tree) == 3:
		print "" + pref[:-width] + top,
		sys.stdout.softspace = False
		print tree[0]
		print_tree(tree[1], "" + pref + son1)
		print_tree(tree[2], "" + pref + son2)
	else:
		print pref[:-width] + leaf,
		sys.stdout.softspace = False
		print tree


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


class PublicKey:
	"""Docstring."""

	def __init__(self, degree=None, modulus=None, hash_to_point=None, signature_bound=None, verify=None):
		"""Docstring."""
		self.n = degree
		self.q = modulus
		self.hash_to_point = hash_to_point
		self.signature_bound = signature_bound
		self.verify = verify


class SecretKey:
	"""
	This class contains methods for performing secret key operations (and also public key operations) in the scheme Falcon.

	One can perform:
	- initializing a secret key for n = 8, 16, 32, 64, 128, 256, 512, 1024, 2048, Phi = x**n+1, q = 12*1024+1
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
		self.hash_function = shake_256

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

		self.T = ffldl(self.G0)
		self.T_fft = ffldl_fft(self.G0_fft)

		"""Private key part 4: compute sigma and signature bound."""
		sq_gs_norm = checknorm(self.f, self.g, q)
		self.sigma = 1.28 * sqrt(sq_gs_norm)
		self.signature_bound = 2 * self.n * (self.sigma**2)

		"""Private key part 5: set leaves of tree to be the standard deviations."""
		normalize_tree(self.T, self.sigma)
		normalize_tree(self.T_fft, self.sigma)

		"""Public key: h such that h*f = g mod (Phi,q)"""
		self.h = div_zq(self.g, self.f)

	def derivate_pk(self):
		"""Generate a public key from the secret key."""
		pk = PublicKey(self.n, self.q, self.hash_to_point, self.signature_bound, self.verify)
		return pk

	def get_coord(self, point):
		"""Compute t such that t*B0 = c."""
		c0, c1 = point
		[[a, b], [c, d]] = self.B0_fft
		c0_fft, c1_fft = fft(c0), fft(c1)
		t0 = ifft([(c0_fft[i] * d[i] - c1_fft[i] * c[i]) / self.q for i in range(self.n)])
		t1 = ifft([(-c0_fft[i] * b[i] + c1_fft[i] * a[i]) / self.q for i in range(self.n)])
		return t0, t1

	def hash_to_point(self, message, salt):
		"""Hash a message to a point in Z[x] mod(Phi, q).

		Inspired by the Parse function from NewHope.
		"""
		n, q = self.n, self.q
		if q > 2**16:
			raise ValueError("The modulus is too large")

		k = (2**16) / q
		hash_instance = self.hash_function()
		hash_instance.update(salt)
		hash_instance.update(message)
		digest = hash_instance.hexdigest(8 * n)  # We take twice the number of bytes that would be needed if there was no rejection
		hashed = [0 for i in range(n)]
		i = 0
		j = 0
		while i < n:
			elt = int(digest[4 * j:4 * (j + 1)], 16)  # Takes 2 bytes, transform them in a 16 bits integer
			if elt < k * q:                      # Implicit rejection sampling
				hashed[i] = elt % q
				i += 1
			j += 1
		return hashed

	def sample_preimage_fft(self, point):
		"""Sample preimage."""
		c = point, [0] * self.n
		t = self.get_coord(c)
		t_fft = [fft(t[0]), fft(t[1])]
		z_fft = ffnp_fft(t_fft, self.T_fft)
		z0 = [int(round(elt)) for elt in ifft(z_fft[0])]
		z1 = [int(round(elt)) for elt in ifft(z_fft[1])]
		z = z0, z1
		v = vecmatmul(z, self.B0)
		s = [sub(c[0], v[0]), sub(c[1], v[1])]
		s = [[int(round(coef)) for coef in elt] for elt in s]
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
		s = self.sample_preimage_fft(hashed)
		"""3. The norm of the signature is checked."""
		norm_sign = sum(sum(elt**2 for elt in part) for part in s)
		if norm_sign < self.signature_bound:
			return r, s
		else:
			print "redo"
			return self.sign(message)

	def verify(self, message, signature):
		"""Verify a signature."""
		r, s = signature
		"""1. hashes a message to a point of Z[x] mod (Phi,q"""
		hashed = self.hash_to_point(message, r)
		"""2. Computes s0 + s1*h."""
		result = add_zq(s[0], mul_zq(s[1], self.h))
		"""3. Verifies that the s0 + s1*h = hashed."""
		# print "h =", self.h
		# print "r =", r
		if any(result[i] != hashed[i] for i in range(self.n)):
			print "The signature does not correspond to the hash!"
			return False
		"""4. Verifies that the norm is small"""
		norm_sign = sum(sum(elt**2 for elt in part) for part in s)
		print "signature bound   = ", self.signature_bound
		print "norm of signature = ", norm_sign
		if norm_sign > self.signature_bound:
			print "The squared norm of the signature is too big:", norm_sign
			return False
		"""5. If the previous steps did not fail, accept."""
		return True
