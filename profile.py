"""If you want to profile your code, you can do as follows:
   - python -m cProfile -o profile/data.pyprof profile.py
   - pyprof2calltree -i profile/data.pyprof -o profile/data.callgrind
"""
from test import *

test_falcon(128, 100)
