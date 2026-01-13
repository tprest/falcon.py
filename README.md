# falcon.py

This repository implements the signature scheme Falcon (https://falcon-sign.info/).
Falcon stands for **FA**st Fourier **L**attice-based **CO**mpact signatures over **N**TRU

**Update 13/01/2026:** The API has been thoroughly changed in order to match the APIs found in other libraries. In particular, verification keys are now bytestrings, and signing keys can be exported from and to bytestrings.

## Content

This repository contains the following files (roughly in order of dependency):

1. [`common.py`](common.py) contains shared functions and constants
1. [`rng.py`](rng.py) implements a ChaCha20-based PRNG, useful for KATs (standalone)
1. [`samplerz.py`](samplerz.py) implements a Gaussian sampler over the integers (standalone)
1. [`fft_constants.py`](fft_constants.py) contains precomputed constants used in the FFT
1. [`ntt_constants.py`](ntt_constants.py) contains precomputed constants used in the NTT
1. [`fft.py`](fft.py) implements the FFT over R[x] / (x<sup>n</sup> + 1)
1. [`ntt.py`](ntt.py) implements the NTT over Z<sub>q</sub>[x] / (x<sup>n</sup> + 1)
1. [`ntrugen.py`](ntrugen.py) generate polynomials f,g,F,G in Z[x] / (x<sup>n</sup> + 1) such that f G - g F = q
1. [`ffsampling.py`](ffsampling.py) implements the fast Fourier sampling algorithm
1. [`falcon.py`](falcon.py) implements Falcon
1. [`test.py`](test.py) implements tests to check that everything is properly implemented


## How to use

1. Import the module: `from falcon import Falcon`
2. Generate a Falcon instance: `falcon = Falcon(n)`
3. Generate a keypair `sk, vk = falcon.keygen()`
4. Now we can sign messages: `sig = falcon.sign(sk, m)`
5. To verify a signature: `falcon.verify(vk, m, sig)`

We also provide functions to import to and from byte strings: `falcon.pack_sk(sk)` and `falcon.unpack_sk(sk_bytes)`.

Upon first use, consider running `make test` to make sure that the code runs properly on your machine. You should obtain (the test battery for `n = 1024` may take a few minutes):

```
python3 test.py
Test Sig KATs       : OK
Test SamplerZ KATs  : OK         (46.887 msec / execution)

Test battery for n = 64
Test FFT            : OK          (0.907 msec / execution)
Test NTT            : OK          (0.957 msec / execution)
Test NTRUGen        : OK        (260.644 msec / execution)
Test ffNP           : OK          (5.024 msec / execution)
Test Compress       : OK          (0.184 msec / execution)
Test Signature      : OK          (6.266 msec / execution)

Test battery for n = 128
Test FFT            : OK          (1.907 msec / execution)
Test NTT            : OK          (2.137 msec / execution)
Test NTRUGen        : OK        (679.113 msec / execution)
Test ffNP           : OK         (11.589 msec / execution)
Test Compress       : OK           (0.36 msec / execution)
Test Signature      : OK         (11.882 msec / execution)

Test battery for n = 256
Test FFT            : OK          (4.298 msec / execution)
Test NTT            : OK          (5.014 msec / execution)
Test NTRUGen        : OK        (778.603 msec / execution)
Test ffNP           : OK         (26.182 msec / execution)
Test Compress       : OK          (0.758 msec / execution)
Test Signature      : OK         (23.865 msec / execution)

Test battery for n = 512
Test FFT            : OK          (9.455 msec / execution)
Test NTT            : OK          (9.997 msec / execution)
Test NTRUGen        : OK       (3578.415 msec / execution)
Test ffNP           : OK         (59.863 msec / execution)
Test Compress       : OK          (1.486 msec / execution)
Test Signature      : OK         (51.545 msec / execution)

Test battery for n = 1024
Test FFT            : OK         (20.706 msec / execution)
Test NTT            : OK         (22.937 msec / execution)
Test NTRUGen        : OK      (17707.189 msec / execution)
Test ffNP           : OK         (135.42 msec / execution)
Test Compress       : OK          (3.292 msec / execution)
Test Signature      : OK        (102.022 msec / execution)
```

## Profiling

I included a makefile target to performing profiling on the code. If you type `make profile` on a Linux machine, you should obtain something along these lines:

![kcachegrind](https://tprest.github.io/images/kcachegrind_falcon.png)

Make sure you have `pyprof2calltree` and `kcachegrind` installed on your machine, or it will not work.


## Author

* **Thomas Prest** (thomas . prest @ pqshield . com)


## Acknowledgements

Thank you to the following people for catching various bugs in the code:
- Dan Middleton
- Nadav Voloch
- Dekel Shiran
- Shlomi Dolev

## Disclaimer

This is not reference code. The reference code of Falcon is on https://falcon-sign.info/. This is work in progress. It is not to be considered secure or suitable for production. Also, I do not guarantee portability on Python 2.x.
However, this Python code is rather simple, so I hope that it will be helpful to people seeking to implement Falcon.

If you find errors or flaw, I will be very happy if you report them to me at the provided address.

## License

MIT
