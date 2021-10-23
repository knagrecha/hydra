from setuptools import setup, find_packages

setup(
    name='hydra',
    version='0.1.0',
    packages=find_packages(include=['hydra', 'hydra.*'])
)