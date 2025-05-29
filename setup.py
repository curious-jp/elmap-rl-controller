from setuptools import find_packages
from distutils.core import setup

setup(
    name='robodog_gym',
    version='1.0.0',
    author='Davide Plozza, Gabriel Margolis',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='dplozza@ethz.ch, gmargo@mit.edu',
    description='Toolkit for deployment of sim-to-real RL controller on Unitree robots.',
    install_requires=['ml_logger==0.8.117',
                      'ml_dash==0.3.25',
                      'jaynes>=0.9.2',
                      'params-proto==2.10.5',
                      'gym>=0.14.0',
                      'tqdm',
                      'matplotlib',
                      'numpy==1.23.5',
                      'setuptools==59.5.0',
                      'imageio',
                      'tensorboard',
                      'roma==1.5.0',
                      'wandb',
                      'pynput'
                      ]
)
