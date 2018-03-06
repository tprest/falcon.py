# falcon.py

This is a private repository implementing the signature scheme Falcon (https://falcon-sign.info/).
Falcon stands for **FA**st Fourier **L**attice-based **CO**mpact signatures over **N**TRU

## Content

This repository contains the following files (in order of dependency):
1. **generate_constants.sage** contains the code which was used to generate the constants used in this project
1. **common.py** contains shared functions and constants
1. **fft_constants.py** contains precomputed constants used in the FFT
1. **ntt_constants.py** contains precomputed constants used in the NTT
1. **fft.py** contains a stand-alone implementation of the FFT over R[x] / (x<sup>n</sup> + 1)
1. **ntt.py** contains a stand-alone implementation of the NTT over Z<sub>q</sub>[x] / (x<sup>n</sup> + 1)
1. **ntrugen.py** generate polynomials f,g,F,G in Z[x] / (x<sup>n</sup> + 1) such that f G - g F = q
1. **sampler.py** implements a Gaussian sampler over the integers
1. **ffsampling.py** implements the fast Fourier sampling algorithm
1. **falcon.py** implements Falcon
1. **test.py** implements tests to check that everything is properly implemented


## How to use

1. Generate a secret key **sk = SecretKey(n)**
1. Generate the corresponding public key **pk = PublicKey(sk)**
1. Now we can sign messages:
   - To plainly sign a message m: **sig = sk.sign(m)**
   - To sign a message m with a pre-chosen 320-bit integer salt: **sig = sk.sign(m, salt)**
1. We can also verify signatures: **pk.verify(m, sig)**

## Todo

- [ ] Compress and decompress
- [ ] Document all the docstrings


## Author

* **Thomas Prest** (thomas.prest@ens.fr)

## Disclaimer

This is work in progress. It is not to be considered suitable for production.
It can, to some extent, be considered reference code, but the "true" reference code of Falcon is on https://falcon-sign.info/.

If you find errors or flaw, I will be very happy if you report them to me at the provided address.

## License

MIT
