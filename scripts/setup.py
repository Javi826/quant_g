# setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="backtest_cy",
    ext_modules=cythonize("backtest_cy.pyx", annotate=True, compiler_directives={'boundscheck': False, 'wraparound': False}),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
