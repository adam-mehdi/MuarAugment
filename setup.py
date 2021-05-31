#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='MuarAugment',
    version='0.1dev',
    description='State-of-the-art data augmentation search algorithms in PyTorch',
    author='Adam Mehdi',
    author_email='amehdi.25@dartmouth.edu',
    url='https://github.com/adam-mehdi/MuarAugment',
    install_requires=['pytorch-lightning', 'kornia'],
    packages=find_packages(),
)

