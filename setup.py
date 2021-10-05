#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='DeepInteract',
    version='1.0.0',
    description='A geometric deep learning pipeline for predicting protein interface contacts.',
    author='Alex Morehead',
    author_email='alex.morehead@gmail.com',
    license='GNU Public License, Version 3.0',
    url='https://github.com/amorehead/DeepInteract',
    install_requires=[
        'setuptools==57.4.0',
        'atom3-py3==0.1.9.8',
        'click==8.0.1',
        'easy-parallel-py3==0.1.6.4',
        'dill==0.3.4',
        'tqdm==4.62.0',
        'Sphinx==4.0.1',
        'torchmetrics==0.5.1',
        'networkx==2.6.2',
        'timm==0.4.12',
        'wandb==0.12.2',
        'pytorch-lightning==1.4.8',
        'fairscale==0.4.0'
    ],
    packages=find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GPL-3.0 Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)
