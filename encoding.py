"""Docstring."""


def compress(v):
    """Docstring."""
    u = ""
    for elt in v:
        s = "1" if elt > 0 else "0"
        s += format((abs(elt) % (1 << 7)), '#09b')[:1:-1]
        s += "0" * (abs(elt) >> 7) + "1"
        u += s
    u += "0" * ((8 - len(u)) % 8)
    return [int(u[8 * i: 8 * i + 8], 2) for i in range(len(u) // 8)]


def decompress(t):
    """Docstring."""
    u = ""
    for elt in t:
        u += bin((1 << 8) ^ elt)[3:]
    v = []
    while u[-1] == "0":
        u = u[:-1]
    while u != "":
        sign = 1 if u[0] == "1" else -1
        low = int(u[7:0:-1], 2)
        i, high = 8, 0
        while u[i] == "0":
            i += 1
            high += 1
        elt = sign * (low + (high << 7))
        v += [elt]
        u = u[i + 1:]
    return v
