"""Docstring."""
from common import q
from fft import sub, mul, div, neg, fft, ifft
from ntt import mul_zq, div_zq
from sampler import sampler_z
from ffsampling import ffldl, ffldl_fft, ffnp, ffnp_fft
from ffsampling import checknorm, gram, sqnorm, vecmatmul
from random import randint, random
from math import pi, sqrt, floor, ceil, exp
from ntrugen import ntru_gen
from falcon import SecretKey, PublicKey


def test_fft(n):
	"""Test the FFT."""
	assert (n in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
	print "n =", n
	for i in range(10):
		f = [randint(-3, 4) for j in range(n)]
		g = [randint(-3, 4) for j in range(n)]
		h = mul(f, g)
		k = div(h, f)
		k = [int(round(elt)) for elt in k]
		if k != g:
			print "(f*g)/f =", k
			print "g =", g
			print "mismatch"
	print "ok"


def test_ntt(n):
	"""Test the NTT."""
	assert (n in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
	print "n =", n
	for i in range(10):
		f = [randint(0, q - 1) for j in range(n)]
		g = [randint(0, q - 1) for j in range(n)]
		h = mul_zq(f, g)
		try:
			k = div_zq(h, f)
			if k != g:
				print "(f*g)/f =", k
				print "g =", g
				print "mismatch"
		except ZeroDivisionError:
			continue
	print "ok"


def gaussian(sigma, mu, x):
	"""Docstring."""
	return exp(- ((x - mu) ** 2) / (2. * (sigma ** 2)))


def test_sampler_z(sigma, mu, n):
	"""Test the integer Gaussian sampler."""
	den = sqrt(2 * pi) * sigma
	start = int(floor(mu - 10 * sigma))
	end = int(ceil(mu + 10 * sigma))
	index = range(start, end)
	ref_table = {z: int(round(n * gaussian(sigma, mu, z)) / den) for z in index}
	obs_table = {z: 0 for z in index}
	for i in range(n):
		z = sampler_z(sigma, mu)
		obs_table[z] += 1
	delta = sum(abs(ref_table[i] - obs_table[i]) for i in index) / float(n)
	print delta


def test_ffnp(n, iterations):
	"""Docstring."""
	print "Testing that:"
	print "1. the two versions (coefficient and FFT embeddings) are consistent"
	print "2. ffnp output lattice vectors close to the targets"

	print "n =", n, ", iterations =", iterations
	assert (n in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]), "Please choose n to be a power-of-two"
	f, g, F, G = ntru_gen(n)
	B = [[g, neg(f)], [G, neg(F)]]
	G0 = gram(B)
	G0_fft = [[fft(elt) for elt in row] for row in G0]
	T = ffldl(G0)
	T_fft = ffldl_fft(G0_fft)

	sqgsnorm = checknorm(f, g, q)
	m = 0
	for i in range(iterations):
		t = [[random() for i in range(n)], [random() for i in range(n)]]
		t_fft = [fft(elt) for elt in t]

		z = ffnp(t, T)
		z_fft = ffnp_fft(t_fft, T_fft)

		zb = [ifft(elt) for elt in z_fft]
		zb = [[round(coef) for coef in elt] for elt in zb]
		if z != zb:
			print "Error: ffnp and ffnp_fft are not consistent"
			return
		diff = [sub(t[0], z[0]), sub(t[1], z[1])]
		diffB = vecmatmul(diff, B)
		norm_zmc = int(round(sqnorm(diffB)))
		m = max(m, norm_zmc)
	th_bound = int((n / 4.) * sqgsnorm)
	if m > th_bound:
		print "Warning: the algorithm does not output vectors as short as it should"
	else:
		print "Test OK\n"


def test_falcon(n, nb_iter):
	"""Docstring."""
	sk = SecretKey(n)
	pk = PublicKey(sk)
	for i in range(nb_iter):
		message = ""
		sig = sk.sign(message)
		if pk.verify(message, sig) is False:
			print "Error"
			return
	print "OK"
	return
