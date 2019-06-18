# -*- coding: utf-8 -*-
# Copyright (C) 2016 Diviyan Kalainathan
# Licence: Apache 2.0

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def setup_package():
    """Install the package."""
    setup(name='gsam',
          version='0.1',
          description='Gumbel Softmax Structural Agnostic Model',
          url='https://github.com/Diviyan-Kalainathan/gSAM',
          author='Diviyan Kalainathan',
          author_email='diviyan.kalainathan@lri.fr',
          license='Apache 2.0',
          packages=['gsam'])


if __name__ == '__main__':
    setup_package()
