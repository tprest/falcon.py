"""
Profile the code with:
> make profile
"""
from test import *

if __name__ == "__main__":
    test_falcon(1024, 100)
    # test_ntrugen(1024, 10)
