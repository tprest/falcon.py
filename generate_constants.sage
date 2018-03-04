"""Generate the complex roots of x ** 2 + 1."""
phi4 = cyclotomic_polynomial(4)
phi4_roots = phi4.complex_roots()
phi4_roots.reverse()

"""Generate the complex roots of x ** n + 1, for n = 4, 8, 16, ..., 1024."""
phi8_roots = sum([[sqrt(elt), - sqrt(elt)] for elt in phi4_roots], [])
phi16_roots = sum([[sqrt(elt), - sqrt(elt)] for elt in phi8_roots], [])
phi32_roots = sum([[sqrt(elt), - sqrt(elt)] for elt in phi16_roots], [])
phi64_roots = sum([[sqrt(elt), - sqrt(elt)] for elt in phi32_roots], [])
phi128_roots = sum([[sqrt(elt), - sqrt(elt)] for elt in phi64_roots], [])
phi256_roots = sum([[sqrt(elt), - sqrt(elt)] for elt in phi128_roots], [])
phi512_roots = sum([[sqrt(elt), - sqrt(elt)] for elt in phi256_roots], [])
phi1024_roots = sum([[sqrt(elt), - sqrt(elt)] for elt in phi512_roots], [])
phi2048_roots = sum([[sqrt(elt), - sqrt(elt)] for elt in phi1024_roots], [])


"""Generate the roots of x ** n + 1 in Z_q,
	for q = 12 * 1024 + 1 and n = 4, 8, 16, ..., 1024."""
q = 12 * 1024 + 1
Zq = Integers(q)
phi4_roots_Zq = [sqrt(R(- 1)), - sqrt(R(- 1))]
phi8_roots_Zq = sum([[sqrt(elt), - sqrt(elt)] for elt in phi4_roots_Zq], [])
phi16_roots_Zq = sum([[sqrt(elt), - sqrt(elt)] for elt in phi8_roots_Zq], [])
phi32_roots_Zq = sum([[sqrt(elt), - sqrt(elt)] for elt in phi16_roots_Zq], [])
phi64_roots_Zq = sum([[sqrt(elt), - sqrt(elt)] for elt in phi32_roots_Zq], [])
phi128_roots_Zq = sum([[sqrt(elt), - sqrt(elt)] for elt in phi64_roots_Zq], [])
phi256_roots_Zq = sum([[sqrt(elt), - sqrt(elt)] for elt in phi128_roots_Zq], [])
phi512_roots_Zq = sum([[sqrt(elt), - sqrt(elt)] for elt in phi256_roots_Zq], [])
phi1024_roots_Zq = sum([[sqrt(elt), - sqrt(elt)] for elt in phi512_roots_Zq], [])
phi2048_roots_Zq = sum([[sqrt(elt), - sqrt(elt)] for elt in phi1024_roots_Zq], [])


RRR = RealField(256)


"""Generate precomputed constants for the Gaussian sampler over Z."""
def gaussian(sigma, mu, x):
	return exp(- (RRR(x) - RRR(mu)) ** 2 / (2 * (RRR(sigma) ** 2)))


def half_gaussian_table(sigma):
	normalization_factor = sum(gaussian(sigma, 0, i) for i in range(10 * sigma))
	table = []
	for i in range(10 * sigma):
		u = sum(gaussian(sigma, 0, j) for j in range(i + 1)) / normalization_factor
		high_u = '0x%016x' % floor(u * (1 << 64))
		low_u = '0x%016x' % (floor(u * (1 << 128)) & ((1 << 64) - 1))
		table += [u]
	return table
