"""
Profile the code with:
> make profile
"""
from test import test_falcon

if __name__ == "__main__":
    test_falcon(128, 100)
