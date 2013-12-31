#!/usr/bin/env python

from setuptools import setup
from pyvectorized import __version__ as pyvectorized_version

setup(name = 'pyvectorized',
      version = pyvectorized_version,
      description = 'Vectorized plotting and numerical utilities',
      author = 'Ioannis Filippidis',
      author_email = 'jfilippidis@gmail.com',
      url = 'https://github.com/johnyf/pyvectorized',
      license = 'BSD',
      requires = ['numpy', 'matplotlib'],
      #install_requires = ['numpy >= 1.6'],
      packages = ['pyvectorized'],
      package_dir = {'pyvectorized':'pyvectorized'},
      #package_data={},
)
