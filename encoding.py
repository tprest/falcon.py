"""SHOULD BE CLARIFIED:12,13,23,25-30"""


"""Docstring."""


def compress(v):
	"""Docstring."""
	u = ""
	for elt in v:
		s = "1" if elt > 0 else "0"
		<???s += format((abs(elt) % (1 << 7)), '#09b')[:1:-1]???>
		<???s += "0" * (abs(elt) >> 7) + "1"???>
		u += s
	return u


def decompress(u):
	"""Docstring."""
	v = []
	while u != "":
		sign = 1 if u[0] == "1" else -1
		<???low = int(u[7:0:-1], 2)???>
		i, high = 8, 0
		<???while u[i] == "0":
			i += 1
			high += 1
		elt = sign * (low + (high << 7))
		v += [elt]
		u = u[i + 1:]???>
	return v
