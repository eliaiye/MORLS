import setuptools
from setuptools import setup

setup(
    name='MORLS',
    version='1.0',
    description='MORLS under developing',
    packages=setuptools.find_packages(),
    install_requires=[
        'torch==1.9.0',
        'numpy', 
        'matplotlib',
        'stable_baselines3',
        'tqdm',
        'pandas',
        'tensorflow==2.5.0',
        'gym==0.18.0',
        'mujoco_py==2.0.2.13'


        ]
    )